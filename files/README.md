# Nova voice prototype (Flutter)

## 1. Create the project, then drop these files in
```
flutter create nova_voice --platforms=android
# replace nova_voice/pubspec.yaml and nova_voice/lib/main.dart with the ones here
cd nova_voice && flutter pub get
```

## 2. Android setup
`android/app/build.gradle(.kts)`: set `minSdk = 26`.

`android/app/src/main/AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
<uses-permission android:name="android.permission.INTERNET"/>
<application android:largeHeap="true" ...>
  <uses-native-library android:name="libOpenCL.so" android:required="false"/>
</application>
```

## 3. Get the brain
Download the Gemma 3 270M instruction-tuned LiteRT file (`.task` or `.litertlm`) from the
`litert-community/gemma-3-270m-it` page on Hugging Face (accept the Gemma license first).
Copy it to the phone, then in the app: swipe left → Settings → Brain → Choose file.

## 4. Get the voice (in the app)
Settings → Voice → Download. One time, about 195 MB, then listening and speaking work offline:
- Hearing: sherpa-onnx streaming Zipformer English 20M (tiny, runs live as you talk)
- Speaking: Piper "Amy" low (small, natural enough, generated on the phone)
Unpacking takes about a minute.

## 5. Use it
- Tap the orb: it glows with your voice while you talk, thinks, then pulses while it speaks.
- Live transcript sits at the bottom of the main screen.
- Swipe from the right edge: Settings. Swipe from the left edge: Shortcuts. Left/right arrow keys also switch pages.
- Settings → Agent → Image or GIF swaps the orb for your own picture (GIFs animate). "Use orb" restores it.

## Notes
- `Agent` in `main.dart` owns the LLM, STT and TTS. Replace `_ask()` to call your Rust core later.
- Shortcuts currently send their action text to the model. Launching real apps comes with the tool layer.
- Speech is generated after the full reply is ready, and generation briefly blocks the UI (short replies are fine). Move it to an isolate if it janks.
- Package APIs (`flutter_gemma`, `sherpa_onnx`, `file_picker`) change between versions. If a call fails to compile, check that package's changelog; the changes needed are small and live in `Agent`.
