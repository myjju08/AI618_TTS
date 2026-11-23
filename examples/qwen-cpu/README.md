# 실제 Qwen CPU 비교 샘플

2026-11-23에 Qwen3-TTS-12Hz-1.7B-CustomVoice, Ryan, CPU float32, eager attention으로 생성한 WAV입니다. 입력 문장은 모두 `The next train will arrive in five minutes.`입니다. 읽을 문장과 스타일 지시는 따로 전달했습니다.

| 상황 | 요청 악센트 | 깨끗한 음성 | 최종 합성 |
| --- | --- | --- | --- |
| Documentary | American | [WAV](clean/documentary-american.wav) | [WAV](mixed/documentary-american.wav) |
| Documentary | British | [WAV](clean/documentary-british.wav) | [WAV](mixed/documentary-british.wav) |
| Station announcement | American | [WAV](clean/station-american.wav) | [WAV](mixed/station-american.wav) |
| Station announcement | British | [WAV](clean/station-british.wav) | [WAV](mixed/station-british.wav) |

Documentary에는 배경음이 없습니다. Station에는 절차적 역 근사음을 10 dB SNR로 합성하고 앞뒤 0.5초 여백을 추가했습니다. 실제 역 녹음은 아닙니다. 파일은 모두 mono 24 kHz float WAV입니다.

생성 성공과 파형 유효성, 합성 SNR은 확인했습니다. **문장 정확성, 톤 차이, 악센트 차이, 지각적인 소음 크기는 청취 검수 전**입니다. American/British 표기는 모델에 요청한 조건입니다.

원본 WAV 바이트를 보존한 채 비교하기 쉬운 파일명으로 복사했습니다. [metadata.csv](metadata.csv)에 프롬프트·seed·환경·경로가, [validation.json](validation.json)에 수치 검증 결과가 있습니다.

```powershell
scenario-tts generate examples/qwen-cpu/sentences.csv --presets documentary station --accents american british --backend qwen --device cpu --max-new-tokens 512 --output outputs/qwen-comparison
```

동일한 seed라도 패키지·모델·하드웨어 차이로 결과가 달라질 수 있습니다. 저장된 WAV는 동작 예시이며 평가용 정답 데이터가 아닙니다.
