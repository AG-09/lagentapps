import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:archive/archive_io.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;
import 'package:shared_preferences/shared_preferences.dart';

// ───────────────────────── Colors ─────────────────────────
class C {
  static const void_ = Color(0xFF000000); // avatar frame
  static const ink = Color(0xFF0D0D0F); // app background
  static const slate = Color(0xFF1A1A1D); // cards
  static const hairline = Color(0xFF2E2E33); // borders
  static const paper = Color(0xFFECECEE); // primary text
  static const mist = Color(0xFF9A9AA3); // secondary text
  static const ember = Color(0xFFD97757); // accent / glow
}

// ───────────────────────── Agent (state + brain + voice) ─────────────────────────
enum Phase { idle, listening, thinking, speaking }

class Shortcut {
  String name, action;
  Shortcut(this.name, this.action);
  Map<String, String> toJson() => {'n': name, 'a': action};
  static Shortcut from(Map m) => Shortcut(m['n'], m['a']);
}

class Agent extends ChangeNotifier {
  late SharedPreferences p;
  late Directory _voiceDir;
  final level = ValueNotifier<double>(0); // live mic level (listening) or voice level (speaking)
  final _rec = AudioRecorder();
  final _player = AudioPlayer();

  Phase phase = Phase.idle;
  String heard = '', reply = '', error = '', modelStatus = 'No model loaded';
  String voiceStatus = 'Voice pack not downloaded';
  double voiceProgress = 0;
  bool downloading = false, voiceReady = false;
  String name = 'Nova';
  String prompt =
      'You are Nova, a fast voice assistant. Reply in one or two short sentences.';
  String? modelPath, avatarPath;
  bool speak = true, haptics = true, reduceMotion = false, modelReady = false;
  double rate = 0.5; // speech speed = rate × 2
  List<Shortcut> shortcuts = [];

  InferenceModel? _model; // ← Rust core plugs in here later
  InferenceChat? _chat;
  sherpa.OnlineRecognizer? _asr; // speech → text (Zipformer 20M, streaming)
  sherpa.OnlineStream? _stream;
  sherpa.OfflineTts? _tts; // text → speech (Piper Amy low)
  StreamSubscription<Uint8List>? _sub;
  Completer<void>? _playing;
  bool _submitted = false, _finishing = false;
  double _lastP = 0;

  Future<void> init() async {
    p = await SharedPreferences.getInstance();
    name = p.getString('name') ?? name;
    prompt = p.getString('prompt') ?? prompt;
    avatarPath = p.getString('avatar');
    modelPath = p.getString('modelPath');
    speak = p.getBool('speak') ?? true;
    haptics = p.getBool('haptics') ?? true;
    reduceMotion = p.getBool('reduce') ?? false;
    rate = p.getDouble('rate') ?? rate;
    final raw = p.getString('shortcuts');
    shortcuts = raw == null
        ? [
            Shortcut('protocol-78', 'Open my notes and start a new note.'),
            Shortcut('protocol-12', 'Start focus mode and silence notifications.'),
            Shortcut('morning-brief', 'Give me a two-sentence summary of my day.'),
          ]
        : (jsonDecode(raw) as List).map((e) => Shortcut.from(e)).toList();

    sherpa.initBindings();
    _voiceDir = Directory('${(await getApplicationSupportDirectory()).path}/voice');
    try {
      await FlutterGemma.initialize();
    } catch (_) {}
    await _loadVoice();
    if (modelPath != null && File(modelPath!).existsSync()) {
      loadModel(modelPath!);
    }
  }

  void update(void Function() f) {
    f();
    p.setString('name', name);
    p.setString('prompt', prompt);
    p.setBool('speak', speak);
    p.setBool('haptics', haptics);
    p.setBool('reduce', reduceMotion);
    p.setDouble('rate', rate);
    p.setString('shortcuts', jsonEncode(shortcuts.map((s) => s.toJson()).toList()));
    if (avatarPath == null) {
      p.remove('avatar');
    } else {
      p.setString('avatar', avatarPath!);
    }
    notifyListeners();
  }

  // ── Brain: load a downloaded Gemma file ──
  Future<void> pickModel() async {
    final r = await FilePicker.platform.pickFiles(type: FileType.any);
    final path = r?.files.single.path;
    if (path != null) await loadModel(path);
  }

  Future<void> loadModel(String path) async {
    modelReady = false;
    modelStatus = 'Loading ${path.split('/').last}…';
    notifyListeners();
    try {
      await FlutterGemma.installModel(modelType: ModelType.gemmaIt)
          .fromFile(path)
          .install();
      _model = await FlutterGemma.getActiveModel(maxTokens: 512);
      modelPath = path;
      await p.setString('modelPath', path);
      modelReady = true;
      modelStatus = 'Ready · ${path.split('/').last}';
    } catch (e) {
      modelStatus = 'Could not load the model. Check the file type. ($e)';
    }
    notifyListeners();
  }

  // ── Avatar ──
  Future<void> pickAvatar() async {
    final r = await FilePicker.platform.pickFiles(type: FileType.image);
    final path = r?.files.single.path;
    if (path != null) update(() => avatarPath = path);
  }

  // ── Voice pack: small offline STT + TTS (sherpa-onnx) ──
  static const _base = 'https://github.com/k2-fsa/sherpa-onnx/releases/download';
  static const _packs = {
    'asr': '$_base/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2',
    'tts': '$_base/tts-models/vits-piper-en_US-amy-low.tar.bz2',
  };

  Future<void> downloadVoices() async {
    if (downloading) return;
    downloading = true;
    voiceReady = false;
    voiceProgress = 0;
    notifyListeners();
    try {
      var i = 0;
      for (final e in _packs.entries) {
        final out = Directory('${_voiceDir.path}/${e.key}');
        if (out.existsSync()) out.deleteSync(recursive: true);
        out.createSync(recursive: true);
        final tar = File('${_voiceDir.path}/${e.key}.tar.bz2');
        final client = http.Client();
        final res = await client.send(http.Request('GET', Uri.parse(e.value)));
        final total = res.contentLength ?? 1;
        var got = 0;
        final sink = tar.openWrite();
        voiceStatus = e.key == 'asr' ? 'Downloading speech recognition…' : 'Downloading voice…';
        await for (final chunk in res.stream) {
          sink.add(chunk);
          got += chunk.length;
          voiceProgress = (i + (got / total).clamp(0.0, 1.0).toDouble() * 0.8) / 2;
          if ((voiceProgress - _lastP).abs() > 0.005) {
            _lastP = voiceProgress;
            notifyListeners();
          }
        }
        await sink.close();
        client.close();
        voiceStatus = 'Unpacking… this takes a minute';
        notifyListeners();
        final src = tar.path, dst = out.path;
        await Isolate.run(() => extractFileToDisk(src, dst));
        tar.deleteSync();
        i++;
        voiceProgress = i / 2;
      }
      downloading = false;
      await _loadVoice();
    } catch (e) {
      downloading = false;
      voiceStatus = 'Download failed. Check your connection and try again. ($e)';
      notifyListeners();
    }
  }

  String? _pick(Directory d, String key) {
    final f = d
        .listSync(recursive: true)
        .whereType<File>()
        .map((f) => f.path)
        .where((x) => x.split('/').last.contains(key) && x.endsWith('.onnx'))
        .toList();
    if (f.isEmpty) return null;
    return f.firstWhere((x) => x.contains('int8'), orElse: () => f.first);
  }

  String? _file(Directory d, bool Function(String) t) => d
      .listSync(recursive: true)
      .whereType<File>()
      .map((f) => f.path)
      .where((x) => t(x.split('/').last))
      .firstOrNull;

  Future<void> _loadVoice() async {
    final asrDir = Directory('${_voiceDir.path}/asr'), ttsDir = Directory('${_voiceDir.path}/tts');
    if (!asrDir.existsSync() || !ttsDir.existsSync()) {
      notifyListeners();
      return;
    }
    try {
      final enc = _pick(asrDir, 'encoder'), dec = _pick(asrDir, 'decoder'), joi = _pick(asrDir, 'joiner');
      final tok = _file(asrDir, (n) => n == 'tokens.txt');
      final vm = _file(ttsDir, (n) => n.endsWith('.onnx'));
      final vt = _file(ttsDir, (n) => n == 'tokens.txt');
      final data = ttsDir
          .listSync(recursive: true)
          .whereType<Directory>()
          .map((d) => d.path)
          .where((x) => x.endsWith('espeak-ng-data'))
          .firstOrNull;
      if ([enc, dec, joi, tok, vm, vt, data].contains(null)) throw 'files missing';
      _asr = sherpa.OnlineRecognizer(sherpa.OnlineRecognizerConfig(
        model: sherpa.OnlineModelConfig(
          transducer: sherpa.OnlineTransducerModelConfig(encoder: enc!, decoder: dec!, joiner: joi!),
          tokens: tok!,
          numThreads: 2,
        ),
        enableEndpoint: true,
        rule2MinTrailingSilence: 1.0,
      ));
      _tts = sherpa.OfflineTts(sherpa.OfflineTtsConfig(
        model: sherpa.OfflineTtsModelConfig(
          vits: sherpa.OfflineTtsVitsModelConfig(model: vm!, tokens: vt!, dataDir: data!),
          numThreads: 2,
        ),
      ));
      voiceReady = true;
      voiceStatus = 'Ready · Zipformer 20M + Piper Amy, fully offline';
    } catch (e) {
      voiceStatus = 'Could not load the voice pack. Try downloading again. ($e)';
    }
    notifyListeners();
  }

  // ── Voice loop: tap → listen → think → speak ──
  void _tick() {
    if (haptics) HapticFeedback.selectionClick();
  }

  Future<void> tap() async {
    _tick();
    switch (phase) {
      case Phase.speaking:
        await _player.stop();
        if (_playing != null && !_playing!.isCompleted) _playing!.complete();
        return;
      case Phase.thinking:
        return;
      case Phase.listening:
        await _finishListening();
        return;
      case Phase.idle:
        break;
    }
    if (!voiceReady) {
      error = 'Download the voice pack in Settings first.';
      notifyListeners();
      return;
    }
    if (!await _rec.hasPermission()) {
      error = 'Allow microphone access to talk.';
      notifyListeners();
      return;
    }
    error = '';
    heard = '';
    reply = '';
    _submitted = false;
    _finishing = false;
    phase = Phase.listening;
    notifyListeners();
    _stream = _asr!.createStream();
    final mic = await _rec.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits, sampleRate: 16000, numChannels: 1));
    _sub = mic.listen(_onAudio);
  }

  void _onAudio(Uint8List bytes) {
    final s = _stream;
    if (s == null || phase != Phase.listening) return;
    final n = bytes.length ~/ 2;
    final f = Float32List(n);
    final bd = ByteData.sublistView(bytes);
    var sum = 0.0;
    for (var i = 0; i < n; i++) {
      final v = bd.getInt16(i * 2, Endian.little) / 32768.0;
      f[i] = v;
      sum += v * v;
    }
    final v = (math.sqrt(sum / math.max(n, 1)) * 7).clamp(0.0, 1.0).toDouble();
    level.value = level.value * 0.5 + v * 0.5;
    s.acceptWaveform(samples: f, sampleRate: 16000);
    while (_asr!.isReady(s)) {
      _asr!.decode(s);
    }
    heard = _asr!.getResult(s).text;
    notifyListeners();
    if (_asr!.isEndpoint(s) && heard.trim().isNotEmpty) _finishListening();
  }

  Future<void> _finishListening() async {
    if (_finishing || phase != Phase.listening) return;
    _finishing = true;
    await _sub?.cancel();
    await _rec.stop();
    final s = _stream;
    if (s != null) {
      s.inputFinished();
      while (_asr!.isReady(s)) {
        _asr!.decode(s);
      }
      heard = _asr!.getResult(s).text;
      s.free();
      _stream = null;
    }
    _submit();
  }

  void _submit() {
    if (_submitted) return;
    _submitted = true;
    level.value = 0;
    final t = heard.trim();
    if (t.isEmpty) {
      phase = Phase.idle;
      notifyListeners();
    } else {
      _ask(t);
    }
  }

  String _norm(String s) => s.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '');

  Future<void> _ask(String said) async {
    phase = Phase.thinking;
    reply = '';
    notifyListeners();
    var q = said;
    for (final s in shortcuts) {
      if (_norm(said).contains(_norm(s.name))) {
        q = s.action; // shortcut hit: run its action instead
        break;
      }
    }
    if (!modelReady) {
      reply = 'Load a model in Settings first.';
      return _say();
    }
    try {
      try {
        await _chat?.close();
      } catch (_) {}
      _chat = await _model!.createChat(temperature: 0.7, topK: 40);
      await _chat!.addQueryChunk(Message.text(text: '$prompt\n\nUser: $q', isUser: true));
      await for (final r in _chat!.generateChatResponseAsync()) {
        if (r is TextResponse) {
          reply += r.token;
          notifyListeners();
        }
      }
    } catch (e) {
      reply = 'Something went wrong while thinking. ($e)';
    }
    await _say();
  }

  Future<void> _say() async {
    if (speak && voiceReady && reply.trim().isNotEmpty) {
      phase = Phase.speaking;
      notifyListeners();
      await _speakText(reply);
    }
    _ttsDone();
  }

  /// Generates speech on-device, then plays it while the orb follows the real loudness.
  Future<void> _speakText(String text) async {
    await Future.delayed(const Duration(milliseconds: 60)); // let the orb repaint first
    final audio = _tts!.generate(
        text: text.replaceAll(RegExp(r'[*#_`]'), ''), sid: 0, speed: (rate * 2).clamp(0.5, 2.0).toDouble());
    final path = '${(await getTemporaryDirectory()).path}/reply.wav';
    sherpa.writeWave(filename: path, samples: audio.samples, sampleRate: audio.sampleRate);

    // loudness envelope, one value per 50 ms
    final w = math.max((audio.sampleRate * 0.05).round(), 1);
    final env = <double>[];
    for (var i = 0; i < audio.samples.length; i += w) {
      final end = math.min(i + w, audio.samples.length);
      var s = 0.0;
      for (var j = i; j < end; j++) {
        s += audio.samples[j] * audio.samples[j];
      }
      env.add(math.sqrt(s / (end - i)));
    }
    final mx = env.fold<double>(0.001, (a, b) => math.max(a, b));
    for (var k = 0; k < env.length; k++) {
      env[k] /= mx;
    }

    _playing = Completer<void>();
    final done = _player.onPlayerComplete.listen((_) {
      if (!_playing!.isCompleted) _playing!.complete();
    });
    final pos = _player.onPositionChanged.listen((d) {
      final i = d.inMilliseconds ~/ 50;
      if (i < env.length) level.value = level.value * 0.4 + env[i] * 0.6;
    });
    await _player.play(DeviceFileSource(path));
    await _playing!.future;
    await done.cancel();
    await pos.cancel();
    level.value = 0;
  }

  void _ttsDone() {
    if (phase != Phase.idle) {
      phase = Phase.idle;
      notifyListeners();
    }
  }

  Future<void> testVoice() async {
    if (!voiceReady || phase != Phase.idle) return;
    phase = Phase.speaking;
    notifyListeners();
    await _speakText('Hi, I am $name. This is how I sound.');
    _ttsDone();
  }
}

// ───────────────────────── App ─────────────────────────
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle.light.copyWith(
    statusBarColor: Colors.transparent,
    systemNavigationBarColor: C.ink,
  ));
  final agent = Agent();
  await agent.init();
  runApp(NovaApp(agent));
}

class NovaApp extends StatelessWidget {
  final Agent a;
  const NovaApp(this.a, {super.key});
  @override
  Widget build(BuildContext context) {
    final base = ThemeData(
      brightness: Brightness.dark,
      useMaterial3: true,
      scaffoldBackgroundColor: C.ink,
      colorScheme: const ColorScheme.dark(
          primary: C.ember, onPrimary: C.void_, surface: C.slate, onSurface: C.paper, outline: C.hairline),
      focusColor: C.ember.withValues(alpha: 0.3),
    );
    return MaterialApp(
      title: 'Nova',
      debugShowCheckedModeBanner: false,
      theme: base.copyWith(
          textTheme: GoogleFonts.interTextTheme(base.textTheme)
              .apply(bodyColor: C.paper, displayColor: C.paper)),
      home: Shell(a),
    );
  }
}

/// Left page = Shortcuts, centre = Main, right page = Settings.
class Shell extends StatefulWidget {
  final Agent a;
  const Shell(this.a, {super.key});
  @override
  State<Shell> createState() => _ShellState();
}

class _ShellState extends State<Shell> {
  final pc = PageController(initialPage: 1);
  void go(int i) => pc.animateToPage(i,
      duration: const Duration(milliseconds: 280), curve: Curves.easeOutCubic);

  @override
  Widget build(BuildContext context) {
    final a = widget.a;
    return CallbackShortcuts(
      bindings: {
        const SingleActivator(LogicalKeyboardKey.arrowLeft): () => go(((pc.page ?? 1).round() - 1).clamp(0, 2).toInt()),
        const SingleActivator(LogicalKeyboardKey.arrowRight): () => go(((pc.page ?? 1).round() + 1).clamp(0, 2).toInt()),
      },
      child: Focus(
        autofocus: true,
        child: Scaffold(
          body: SafeArea(
            child: PageView(controller: pc, children: [
              ShortcutsPage(a),
              MainPage(a),
              SettingsPage(a),
            ]),
          ),
        ),
      ),
    );
  }
}

// ───────────────────────── Main screen ─────────────────────────
class MainPage extends StatelessWidget {
  final Agent a;
  const MainPage(this.a, {super.key});

  String get status => switch (a.phase) {
        Phase.idle => a.error.isNotEmpty ? a.error : 'Tap to talk',
        Phase.listening => 'Listening',
        Phase.thinking => 'Thinking',
        Phase.speaking => 'Speaking · tap to stop',
      };

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: a,
      builder: (context, _) => LayoutBuilder(builder: (context, box) {
        final wide = box.maxWidth > box.maxHeight * 1.1;
        final orbSize = wide
            ? math.min(box.maxHeight * 0.85, box.maxWidth * 0.42)
            : math.min(box.maxWidth * 0.85, box.maxHeight * 0.46);
        final identity = Column(mainAxisSize: MainAxisSize.min, children: [
          Orb(a, size: orbSize),
          Text(a.name,
              style: GoogleFonts.sourceSerif4(fontSize: 30, fontWeight: FontWeight.w600, color: C.paper)),
          const SizedBox(height: 4),
          Text(status,
              textAlign: TextAlign.center,
              style: const TextStyle(color: C.mist, fontSize: 14)),
        ]);
        final card = TranscriptCard(a);
        if (wide) {
          return Padding(
            padding: const EdgeInsets.all(20),
            child: Row(children: [
              Expanded(child: Center(child: identity)),
              const SizedBox(width: 20),
              Expanded(child: Align(alignment: Alignment.bottomCenter, child: card)),
            ]),
          );
        }
        return Padding(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
          child: Column(children: [
            Expanded(child: Center(child: identity)),
            card,
          ]),
        );
      }),
    );
  }
}

class TranscriptCard extends StatelessWidget {
  final Agent a;
  const TranscriptCard(this.a, {super.key});

  Widget line(String who, String text, {bool dim = false}) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 3),
        child: Text.rich(TextSpan(children: [
          TextSpan(
              text: '$who  ',
              style: const TextStyle(color: C.mist, fontSize: 12, fontWeight: FontWeight.w600)),
          TextSpan(text: text, style: TextStyle(color: dim ? C.mist : C.paper, fontSize: 15, height: 1.35)),
        ])),
      );

  @override
  Widget build(BuildContext context) {
    return Semantics(
      liveRegion: true,
      label: 'Live transcript',
      child: Container(
        width: double.infinity,
        constraints: const BoxConstraints(maxHeight: 132, maxWidth: 560),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: C.slate,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: C.hairline),
        ),
        child: SingleChildScrollView(
          reverse: true,
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            line('You', a.heard.isEmpty ? 'Your words appear here as you speak.' : a.heard,
                dim: a.heard.isEmpty),
            if (a.reply.isNotEmpty) line(a.name, a.reply),
          ]),
        ),
      ),
    );
  }
}

// ───────────────────────── Orb / avatar ─────────────────────────
class Orb extends StatefulWidget {
  final Agent a;
  final double size;
  const Orb(this.a, {super.key, required this.size});
  @override
  State<Orb> createState() => _OrbState();
}

class _OrbState extends State<Orb> with SingleTickerProviderStateMixin {
  late final AnimationController c =
      AnimationController(vsync: this, duration: const Duration(seconds: 8))..repeat();
  bool focused = false;

  @override
  void dispose() {
    c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final a = widget.a;
    final path = a.avatarPath;
    final hasImg = path != null && File(path).existsSync();
    final fr = widget.size / 2 * 0.74;
    return Semantics(
      button: true,
      label: '${a.name}. ${a.phase.name}. Double tap to talk.',
      child: FocusableActionDetector(
        onShowFocusHighlight: (v) => setState(() => focused = v),
        actions: {
          ActivateIntent: CallbackAction<ActivateIntent>(onInvoke: (_) {
            a.tap();
            return null;
          }),
        },
        child: GestureDetector(
          onTap: a.tap,
          child: SizedBox(
            width: widget.size,
            height: widget.size,
            child: AnimatedBuilder(
              animation: Listenable.merge([c, a.level]),
              builder: (context, _) => Stack(alignment: Alignment.center, children: [
                CustomPaint(
                  size: Size.square(widget.size),
                  painter: _OrbPainter(
                      phase: a.phase,
                      level: a.level.value,
                      t: c.value,
                      focused: focused,
                      hasImage: hasImg,
                      reduce: a.reduceMotion),
                ),
                if (hasImg)
                  ClipOval(
                    child: Image.file(File(path), // GIFs animate automatically
                        width: fr * 2 - 8, height: fr * 2 - 8, fit: BoxFit.cover, gaplessPlayback: true),
                  ),
              ]),
            ),
          ),
        ),
      ),
    );
  }
}

class _OrbPainter extends CustomPainter {
  final Phase phase;
  final double level, t;
  final bool focused, hasImage, reduce;
  _OrbPainter(
      {required this.phase,
      required this.level,
      required this.t,
      required this.focused,
      required this.hasImage,
      required this.reduce});

  @override
  void paint(Canvas cv, Size s) {
    final ctr = s.center(Offset.zero);
    final R = s.width / 2, fr = R * 0.74;
    final ph = t * 2 * math.pi;

    // glow intensity 0..1: follows the mic while you talk and the voice while it answers
    final g = switch (phase) {
      Phase.listening => 0.25 + 0.75 * level,
      Phase.speaking => reduce ? 0.7 : 0.3 + 0.7 * level,
      Phase.thinking => reduce ? 0.45 : 0.35 + 0.15 * math.sin(ph * 2),
      Phase.idle => reduce ? 0.15 : 0.15 + 0.07 * math.sin(ph),
    };

    // halo
    cv.drawCircle(
        ctr,
        fr * (1.04 + 0.14 * g),
        Paint()
          ..color = C.ember.withValues(alpha: 0.6 * g)
          ..maskFilter = MaskFilter.blur(BlurStyle.normal, R * 0.14));

    // ripples
    if (!reduce && (phase == Phase.listening || phase == Phase.speaking)) {
      final speed = phase == Phase.listening ? 2 : 4;
      for (var i = 0; i < 2; i++) {
        final k = (t * speed + i * 0.5) % 1;
        cv.drawCircle(
            ctr,
            fr * (1.02 + 0.28 * k),
            Paint()
              ..style = PaintingStyle.stroke
              ..strokeWidth = 2
              ..color = C.ember.withValues(alpha: (1 - k) * 0.5 * g));
      }
    }

    // frame
    cv.drawCircle(ctr, fr, Paint()..color = C.void_);
    cv.drawCircle(
        ctr,
        fr,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = focused ? 4 : 2
          ..color = focused ? C.ember : C.ember.withValues(alpha: 0.25 + 0.5 * g));

    if (hasImage) return;

    // core orb with organic wobble
    final r0 = fr * 0.42 * (1 + 0.18 * g);
    final path = Path();
    for (var i = 0; i <= 72; i++) {
      final ang = i / 72 * 2 * math.pi;
      final w = reduce ? 0.0 : g * (0.05 * math.sin(ang * 3 + ph * 2) + 0.04 * math.sin(ang * 5 - ph * 3));
      final p = ctr + Offset.fromDirection(ang, r0 * (1 + w));
      i == 0 ? path.moveTo(p.dx, p.dy) : path.lineTo(p.dx, p.dy);
    }
    cv.drawPath(
        path,
        Paint()
          ..color = C.ember.withValues(alpha: 0.55)
          ..maskFilter = MaskFilter.blur(BlurStyle.normal, r0 * 0.35));
    cv.drawPath(path, Paint()..color = C.ember);
    cv.drawCircle(ctr + Offset(-r0 * 0.18, -r0 * 0.2), r0 * 0.42,
        Paint()..color = const Color(0xFFF2A488).withValues(alpha: 0.8));

    // thinking: rotating arc
    if (phase == Phase.thinking) {
      cv.drawArc(
          Rect.fromCircle(center: ctr, radius: fr * 0.72),
          reduce ? 0 : ph,
          math.pi * 0.6,
          false,
          Paint()
            ..style = PaintingStyle.stroke
            ..strokeWidth = 3
            ..strokeCap = StrokeCap.round
            ..color = C.ember.withValues(alpha: 0.85));
    }
  }

  @override
  bool shouldRepaint(covariant _OrbPainter o) => true;
}

// ───────────────────────── Settings ─────────────────────────
Widget section(String title, List<Widget> kids) => Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(4, 0, 4, 8),
          child: Text(title, style: const TextStyle(color: C.mist, fontSize: 13, fontWeight: FontWeight.w600)),
        ),
        Container(
          decoration: BoxDecoration(
              color: C.slate, borderRadius: BorderRadius.circular(16), border: Border.all(color: C.hairline)),
          child: Column(children: kids),
        ),
      ]),
    );

class SettingsPage extends StatelessWidget {
  final Agent a;
  const SettingsPage(this.a, {super.key});

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: a,
      builder: (context, _) => ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Text('Settings', style: GoogleFonts.sourceSerif4(fontSize: 28, fontWeight: FontWeight.w600)),
          const SizedBox(height: 16),
          section('Brain', [
            ListTile(
              minTileHeight: 64,
              title: const Text('Local model'),
              subtitle: Text(a.modelStatus),
              trailing: FilledButton(onPressed: a.pickModel, child: const Text('Choose file')),
            ),
            const Padding(
              padding: EdgeInsets.fromLTRB(16, 0, 16, 14),
              child: Text('Pick a downloaded Gemma 3 270M file (.task or .litertlm). It runs fully on this phone.',
                  style: TextStyle(color: C.mist, fontSize: 13)),
            ),
          ]),
          section('Agent', [
            Padding(
              padding: const EdgeInsets.all(16),
              child: TextFormField(
                initialValue: a.name,
                decoration: const InputDecoration(labelText: 'Name', border: OutlineInputBorder()),
                onChanged: (v) => a.update(() => a.name = v.isEmpty ? 'Nova' : v),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: TextFormField(
                initialValue: a.prompt,
                minLines: 2,
                maxLines: 5,
                decoration: const InputDecoration(labelText: 'Instructions', border: OutlineInputBorder()),
                onChanged: (v) => a.update(() => a.prompt = v),
              ),
            ),
            ListTile(
              minTileHeight: 64,
              title: const Text('Profile picture'),
              subtitle: Text(a.avatarPath == null ? 'Animated orb' : a.avatarPath!.split('/').last,
                  maxLines: 1, overflow: TextOverflow.ellipsis),
              trailing: Wrap(spacing: 8, children: [
                if (a.avatarPath != null)
                  TextButton(onPressed: () => a.update(() => a.avatarPath = null), child: const Text('Use orb')),
                FilledButton(onPressed: a.pickAvatar, child: const Text('Image or GIF')),
              ]),
            ),
          ]),
          section('Voice', [
            ListTile(
              minTileHeight: 64,
              title: const Text('Voice pack'),
              subtitle: Text(a.voiceStatus),
              trailing: a.downloading
                  ? null
                  : FilledButton(
                      onPressed: a.downloadVoices, child: Text(a.voiceReady ? 'Re-download' : 'Download')),
            ),
            if (a.downloading)
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
                child: LinearProgressIndicator(value: a.voiceProgress, semanticsLabel: 'Voice pack download'),
              ),
            const Padding(
              padding: EdgeInsets.fromLTRB(16, 0, 16, 14),
              child: Text('About 195 MB, once. After that, listening and speaking work offline.',
                  style: TextStyle(color: C.mist, fontSize: 13)),
            ),
            SwitchListTile(
              title: const Text('Speak replies'),
              value: a.speak,
              onChanged: (v) => a.update(() => a.speak = v),
            ),
            ListTile(
              title: const Text('Speech speed'),
              subtitle: Slider(
                  value: a.rate, min: 0.3, max: 1.0, divisions: 7,
                  label: '${(a.rate * 2).toStringAsFixed(1)}×',
                  onChanged: (v) => a.update(() => a.rate = v)),
            ),
            ListTile(
              minTileHeight: 56,
              title: const Text('Hear a sample'),
              trailing: OutlinedButton(onPressed: a.voiceReady ? a.testVoice : null, child: const Text('Play')),
            ),
          ]),
          section('Feel', [
            SwitchListTile(
              title: const Text('Haptics'),
              value: a.haptics,
              onChanged: (v) => a.update(() => a.haptics = v),
            ),
            SwitchListTile(
              title: const Text('Reduce motion'),
              subtitle: const Text('Keeps the orb steady instead of pulsing'),
              value: a.reduceMotion,
              onChanged: (v) => a.update(() => a.reduceMotion = v),
            ),
          ]),
          section('Privacy', [
            const ListTile(
              title: Text('On-device only'),
              subtitle: Text('Voice, text, and the model stay on this phone.'),
            ),
            ListTile(
              minTileHeight: 56,
              title: const Text('Clear conversation'),
              trailing: OutlinedButton(
                  onPressed: () => a.update(() {
                        a.heard = '';
                        a.reply = '';
                      }),
                  child: const Text('Clear')),
            ),
          ]),
        ],
      ),
    );
  }
}

// ───────────────────────── Shortcuts ─────────────────────────
class ShortcutsPage extends StatelessWidget {
  final Agent a;
  const ShortcutsPage(this.a, {super.key});

  Future<void> edit(BuildContext context, [int? i]) async {
    final n = TextEditingController(text: i == null ? '' : a.shortcuts[i].name);
    final act = TextEditingController(text: i == null ? '' : a.shortcuts[i].action);
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(i == null ? 'New shortcut' : 'Edit shortcut'),
        content: Column(mainAxisSize: MainAxisSize.min, children: [
          TextField(
              controller: n,
              autofocus: true,
              style: GoogleFonts.jetBrainsMono(),
              decoration: const InputDecoration(labelText: 'Say this', hintText: 'protocol-78')),
          const SizedBox(height: 12),
          TextField(
              controller: act,
              minLines: 2,
              maxLines: 4,
              decoration: const InputDecoration(labelText: 'Do this', hintText: 'Open my notes and start a new note')),
        ]),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Save shortcut')),
        ],
      ),
    );
    if (ok != true || n.text.trim().isEmpty || act.text.trim().isEmpty) return;
    a.update(() {
      final s = Shortcut(n.text.trim(), act.text.trim());
      i == null ? a.shortcuts.add(s) : a.shortcuts[i] = s;
    });
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: a,
      builder: (context, _) => ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Row(children: [
            Expanded(child: Text('Shortcuts', style: GoogleFonts.sourceSerif4(fontSize: 28, fontWeight: FontWeight.w600))),
            IconButton.filled(
              tooltip: 'Add shortcut',
              onPressed: () => edit(context),
              icon: const Icon(Icons.add),
            ),
          ]),
          const Padding(
            padding: EdgeInsets.only(top: 4, bottom: 16),
            child: Text('Say a shortcut name and the agent runs its action.', style: TextStyle(color: C.mist)),
          ),
          if (a.shortcuts.isEmpty)
            const Padding(
              padding: EdgeInsets.all(24),
              child: Text('No shortcuts yet. Tap + to add one, like “protocol-78”.', style: TextStyle(color: C.mist)),
            ),
          for (var i = 0; i < a.shortcuts.length; i++)
            Card(
              color: C.slate,
              margin: const EdgeInsets.only(bottom: 10),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16), side: const BorderSide(color: C.hairline)),
              child: ListTile(
                minTileHeight: 64,
                onTap: () => edit(context, i),
                title: Text(a.shortcuts[i].name, style: GoogleFonts.jetBrainsMono(color: C.ember, fontSize: 15)),
                subtitle: Text(a.shortcuts[i].action),
                trailing: IconButton(
                  tooltip: 'Delete ${a.shortcuts[i].name}',
                  icon: const Icon(Icons.delete_outline),
                  onPressed: () => a.update(() => a.shortcuts.removeAt(i)),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
