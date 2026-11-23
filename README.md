# Scenario TTS

**정해진 영어 문장을 여러 상황에 맞춰 연출하는 배치 TTS 생성기**입니다. 문장 `text`를 유지한 채, 상황 preset을 발화 지시 `instruct`와 배경음 설정으로 바꿉니다. 모델을 재학습하거나 LLM으로 지시를 생성하지 않습니다.

기본 입력은 **문장 5개 × 상황 6개 × 요청 악센트 2개 = 60개**입니다. Qwen3-TTS 1.7B CustomVoice의 고정 화자 Ryan을 기본으로 사용하며 Aiden도 선택할 수 있습니다.

## 구현 범위

| 상황 | 발화 지시의 의도 | 기본 배경 |
| --- | --- | --- |
| Documentary | 절제된 감정, 일정한 속도의 다큐 내레이션 | 없음 |
| Café conversation | 친구에게 편한 억양으로 말하기 | 카페 근사음 |
| Station announcement | 안내 방송처럼 또렷하게 정보 전달 | 역 근사음 |
| Urgent scene | 친구에게 서둘러야 한다고 긴박하지만 명료하게 말하기 | 거리 근사음 |
| Reassuring scene | 불안한 상대를 부드럽게 안심시키기 | 실내 근사음 |
| Excited scene | 좋은 소식을 밝고 들뜬 목소리로 전달 | 없음 |

- American / British 요청, 톤 덮어쓰기, 화자·배경·SNR·seed 설정
- CSV 배치 생성, 원음 / 합성 WAV 비교, 생성 정보 CSV 저장
- 배경의 0.5초 앞뒤 여백, 페이드, seed에 따른 시작 위치, SNR 20 / 10 / 5 dB
- 샘플 하나만 새 seed로 재생성하고 기존 파일 유지
- 문장 정확성 / 톤 구분 / 악센트 / 소음의 네 항목 수동 검수
- 모델이 필요 없는 **Demo 비음성 신호**로 저장·합성·배치 처리 확인

## 설치

Python 3.10 이상, 3.14 미만을 사용하세요. Qwen 공식 가이드는 새 Python 3.12 환경을 권장합니다. 아래는 [uv](https://docs.astral.sh/uv/) 기준이며, 일반 `python -m venv`와 `pip`로도 설치할 수 있습니다.

```powershell
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Windows PowerShell에서는 `.\.venv\Scripts\Activate.ps1`, macOS/Linux에서는 `source .venv/bin/activate`로 가상환경을 활성화합니다. 활성화하지 않으면 `uv run --no-sync`를 명령 앞에 붙이세요.

실제 TTS 생성에는 추가 의존성이 필요합니다.

```powershell
uv pip install -e ".[tts]"
```

저장소의 `uv.lock`에 기록된 버전으로 설치하려면 다음 명령을 사용할 수 있습니다. 실제 TTS 없이 테스트만 할 때는 `--extra tts`를 빼세요.

```powershell
uv sync --frozen --extra dev --extra tts
uv run --no-sync python app.py
```

GPU를 쓰려면 해당 시스템에 맞는 CUDA용 PyTorch가 필요합니다. [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)를 확인하세요. GPU가 없으면 `cpu`로 실행할 수 있으나 1.7B 모델의 생성은 느립니다. 최초 실행은 Hugging Face에서 모델과 토크나이저를 내려받으므로 네트워크, 저장 공간과 충분한 메모리가 필요합니다. FlashAttention 설치는 필수가 아닙니다.

## 화면 실행

```powershell
python app.py
```

브라우저에서 `http://127.0.0.1:7860`을 엽니다. 로컬 주소로만 실행하며 공개 공유 링크를 만들지 않습니다.

1. 예제 문장 또는 `sentence_id,text` CSV를 선택합니다.
2. 상황과 요청 악센트를 선택하고 **조합과 프롬프트 미리보기**를 누릅니다.
3. 필요한 경우 세부 설정에서 톤·배경·SNR·화자·seed를 수정합니다.
4. **선택한 조합 생성**을 누르고 원음과 합성음을 비교합니다.
5. 마음에 들지 않는 샘플만 재생성하거나 네 가지 검수 항목을 저장합니다.

먼저 `Demo · 비음성 테스트 신호`로 프로그램의 전체 흐름을 확인할 수 있습니다. **Demo는 문장을 읽지 않으며 톤이나 악센트를 구현하지 않습니다.** 실제 음성 품질은 Qwen 결과로 판단해야 합니다.

## CLI 실행

```powershell
# 60개 조합과 지시만 계획합니다. 모델은 로드하지 않습니다.
scenario-tts plan data/sentences.csv --output outputs/plan.csv

# GPU/모델 없이 60개의 테스트 신호를 생성합니다.
scenario-tts generate data/sentences.csv --backend demo --output outputs/demo

# 실제 Qwen 음성: 처음에는 작은 조합부터 확인하세요.
scenario-tts generate data/sentences.csv --backend qwen --presets documentary --accents american --output outputs/qwen-smoke

# 60개 전체 조합
scenario-tts generate data/sentences.csv --backend qwen --output outputs/qwen-full

# 직접 준비한 배경음으로 합성
scenario-tts generate data/sentences.csv --backend qwen --presets cafe station --accents american british --background-dir assets/backgrounds --snr-db 10 --output outputs/custom-backgrounds
```

`--device cpu` 또는 `--device cuda:0`, `--speaker Aiden`, `--seed 123`, `--tone "Speak warmly and slowly."`, `--background room`으로 설정을 조절합니다. 자세한 옵션은 `scenario-tts --help`와 각 하위 명령의 `--help`에서 확인할 수 있습니다.

CSV의 ID는 문자열로 유지하므로 `001`이 `1`로 바뀌지 않습니다. ID 중복과 빈 문장은 오류로 처리합니다.

```csv
sentence_id,text
001,The next train will arrive in five minutes.
002,Please leave your bag beside the door.
```

## 결과 파일

UI는 실행별로 `outputs/날짜_고유값/` 폴더를 만듭니다. CLI는 `--output`으로 폴더를 지정합니다.

```text
outputs/<run>/
  clean/          # 배경이 없는 원음 WAV
  mixed/          # 배경을 합성한 WAV
  metadata.csv    # 요청 조건, seed, 모델, 상태, 파일 경로, 음향 설정
  reviews.csv     # UI에서 저장한 수동 검수 (선택)
```

생성 정보에는 원문, 상황, 요청 악센트, 화자, 실제 instruct, seed, backend, 모델 ID, SNR과 배경 출처가 기록됩니다. 일부 샘플이 실패하면 오류를 기록하고 나머지를 계속 처리합니다. 같은 CLI 출력 폴더에서 동일한 작업을 다시 실행하면 유효한 기존 결과를 재사용합니다. 새 seed로 재생성하면 기존 샘플을 보존합니다.

## 배경음과 SNR

기본 배경음은 별도 다운로드가 없는 **절차적 근사음**입니다. 실제 카페·역·거리 녹음을 제공하는 것은 아닙니다. 직접 준비한 WAV로 교체하는 방법은 [배경음 안내](assets/backgrounds/README.md)를 보세요.

SNR은 발화 구간의 음성 RMS와 배경 RMS 비율입니다. `20 dB`는 약한 배경, `10 dB`는 보통, `5 dB`는 강한 배경의 시작값입니다. 실제 장소의 소음 수준이나 사람의 지각적 시끄러움을 측정한 수치가 아닙니다. 두 신호를 함께 감쇠해 클리핑을 막고 상대 비율은 유지합니다.

## 품질 검수와 한계

1. **문장이 정확한가?** 누락·반복·추가 단어가 없는지 원문과 비교합니다.
2. **톤이 구별되는가?** 같은 문장의 여러 상황을 비교합니다.
3. **요청 악센트가 들리는가?** American/British 지시를 넣었다는 사실만으로 차이가 보장되지는 않습니다.
4. **소음이 과하지 않은가?** 배경을 넣어도 내용을 알아들을 수 있는지 확인합니다.

Qwen은 고정 화자에 자연어 스타일 지시를 전달할 수 있지만, 각 화자가 요청한 악센트를 안정적으로 구현하는지는 별도 청취 검증이 필요합니다. 본 프로젝트는 악센트 제어 정확도나 동일 화자의 악센트 변환을 보장하지 않습니다. 향후 reference voice를 사용한다면 **악센트별 화자 preset**으로 구분해야 하며 현재 버전에는 포함하지 않았습니다.

seed는 생성 조건을 기록하고 같은 환경에서 재현을 시도하기 위한 값입니다. PyTorch·모델 버전·하드웨어가 달라지면 완전히 같은 WAV가 나오지 않을 수 있습니다. 자동화 테스트는 코드 경로와 수치 처리를 확인하며 청취 품질을 증명하지 않습니다.

## 테스트

```powershell
pytest -q
```

단위 테스트는 모델 다운로드 없이 CSV 검증, 조합 생성, 실제 TTS 호출 인자, 실패 처리, SNR과 파일 저장 등을 확인합니다. 실제 모델 검증은 별도로 작은 Qwen 배치를 생성해 듣고 기록하세요.

Qwen 연동은 [공식 Qwen3-TTS 문서](https://github.com/QwenLM/Qwen3-TTS)의 `generate_custom_voice(text=..., language="English", speaker=..., instruct=...)` 인터페이스를 사용합니다.
