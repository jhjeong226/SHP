# CRNP SHP Calibration System

CRNP(Cosmic-Ray Neutron Probe) 기반 토양수분 교정 방법 비교 시스템.

## 교정 방법

| 방법 | 설명 | 파라미터 |
|------|------|----------|
| Standard | N0만 최적화, Wlat 고정 | N0 |
| UTS | Universal Transfer Standard | N0 |
| SHP-Joint | N0 + a2 동시 RMSE 최적화 | N0, a2 |
| SHP-2pt | 건조/습윤 2점 해석적 풀이 | N0, a2 |

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

```bash
# 단일 관측소
python scripts/run_pipeline.py --station HC

# 다중 관측소
python scripts/run_pipeline.py --station HC PC LC

# 방법 비교
python scripts/compare_methods.py --station HC
```

## 구조

```
shp_crnp/
├── config/stations/    # 관측소별 YAML 설정
├── data/input/         # 원시 데이터 (FDR CSV, CRNP Excel)
├── data/output/        # 전처리 결과
├── results/            # 교정 결과 (방법별)
├── src/
│   ├── preprocessing/  # FDRProcessor, CRNPProcessor
│   ├── calibration/    # BaseCalibrator + 각 방법
│   ├── calculation/    # VWCCalculator
│   └── utils/          # io, metrics, plotting
└── scripts/            # 실행 스크립트
```