# 실제 Qwen 음성 16개 비교

2026-11-24 자동 검수에 사용한 Qwen3-TTS 1.7B CustomVoice / Ryan / CPU 결과입니다. 기존 4개를 재사용하고 12개를 추가 생성했습니다. 원본 WAV 바이트를 보존하고 비교하기 쉬운 파일명으로 복사했습니다.

[독립 ASR·음향 수치 분석](../../reports/quality-audit-2026-11-24/README.md)에서 원음과 최종 합성본 각각 16/16개의 전사가 정규화한 원문과 일치했습니다. **자동 인식 결과이며 사람이 직접 듣고 평가한 결과는 아닙니다.** 요청 악센트의 정확성, 톤의 적절성, 자연스러움은 검증하지 않았습니다.

| 문장 ID | 상황 | 요청 악센트 | 원음 | 합성본 | 배경 / SNR |
| --- | --- | --- | --- | --- | --- |
| 001 | cafe | american | [WAV](clean/001-cafe-american.wav) | [WAV](mixed/001-cafe-american.wav) | 카페 근사음 / 20.0 dB |
| 001 | cafe | british | [WAV](clean/001-cafe-british.wav) | [WAV](mixed/001-cafe-british.wav) | 카페 근사음 / 20.0 dB |
| 001 | documentary | american | [WAV](clean/001-documentary-american.wav) | [WAV](mixed/001-documentary-american.wav) | 없음 |
| 001 | documentary | british | [WAV](clean/001-documentary-british.wav) | [WAV](mixed/001-documentary-british.wav) | 없음 |
| 001 | excited | american | [WAV](clean/001-excited-american.wav) | [WAV](mixed/001-excited-american.wav) | 없음 |
| 001 | excited | british | [WAV](clean/001-excited-british.wav) | [WAV](mixed/001-excited-british.wav) | 없음 |
| 001 | reassuring | american | [WAV](clean/001-reassuring-american.wav) | [WAV](mixed/001-reassuring-american.wav) | 실내 근사음 / 20.0 dB |
| 001 | reassuring | british | [WAV](clean/001-reassuring-british.wav) | [WAV](mixed/001-reassuring-british.wav) | 실내 근사음 / 20.0 dB |
| 001 | station | american | [WAV](clean/001-station-american.wav) | [WAV](mixed/001-station-american.wav) | 역 근사음 / 10.0 dB |
| 001 | station | british | [WAV](clean/001-station-british.wav) | [WAV](mixed/001-station-british.wav) | 역 근사음 / 10.0 dB |
| 001 | urgent | american | [WAV](clean/001-urgent-american.wav) | [WAV](mixed/001-urgent-american.wav) | 거리 근사음 / 10.0 dB |
| 001 | urgent | british | [WAV](clean/001-urgent-british.wav) | [WAV](mixed/001-urgent-british.wav) | 거리 근사음 / 10.0 dB |
| 002 | documentary | american | [WAV](clean/002-documentary-american.wav) | [WAV](mixed/002-documentary-american.wav) | 없음 |
| 003 | documentary | american | [WAV](clean/003-documentary-american.wav) | [WAV](mixed/003-documentary-american.wav) | 없음 |
| 004 | documentary | american | [WAV](clean/004-documentary-american.wav) | [WAV](mixed/004-documentary-american.wav) | 없음 |
| 005 | documentary | american | [WAV](clean/005-documentary-american.wav) | [WAV](mixed/005-documentary-american.wav) | 없음 |

## 입력 문장

| ID | text |
| --- | --- |
| 001 | The next train will arrive in five minutes. |
| 002 | Please leave your bag beside the door. |
| 003 | We have enough time to get there together. |
| 004 | I found the place you were looking for. |
| 005 | Your friends are waiting just around the corner. |

[metadata.csv](metadata.csv)에 원문·프롬프트·seed·생성 환경과 음향 설정을 기록했습니다. 배경이 없는 조건에서는 원음과 합성본의 내용이 같고, 배경이 있는 조건에는 앞뒤 0.5초의 여백을 추가했습니다. 모든 배경은 절차적으로 합성한 근사음이며 실제 장소 녹음이 아닙니다.

## 같은 조건으로 다시 생성

아래 두 명령은 같은 출력 폴더를 사용합니다. 두 번째 명령의 `001 / documentary / american`은 첫 번째 명령의 기존 결과를 재사용하므로 총 16개의 고유 조건이 남습니다. 모델·패키지·하드웨어 차이로 같은 seed에서도 결과가 달라질 수 있습니다.

```powershell
$env:OMP_NUM_THREADS="4"
$env:MKL_NUM_THREADS="4"
scenario-tts generate examples/qwen-cpu/sentences.csv --presets documentary cafe station urgent reassuring excited --accents american british --backend qwen --device cpu --max-new-tokens 512 --output outputs/quality-audit
scenario-tts generate data/sentences.csv --presets documentary --accents american --backend qwen --device cpu --max-new-tokens 512 --output outputs/quality-audit
```
