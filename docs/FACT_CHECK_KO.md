# 프로젝트 팩트체크 기록

확인일: 2026-10-11. 보고서의 수치와 구현을 저장된 실험 산출물에 대조하고, 참고문헌은 공식 저장소·출판사 등록 정보·ACL Anthology·버전별 API 문서에서 확인했다.

## 참고문헌 확인

| 항목 | 확인 결과와 근거 |
| --- | --- |
| 과제 안내 | 제공된 과제 PDF에서 제목, 수정일, 제출 요구사항을 확인. 공개 논문으로 표시하지 않음. |
| AlpacaEval | 공식 BibTeX의 저자 8명·제목·2023년을 확인하여 보고서의 설명형 인용을 정식 서지정보로 교체. [고정 revision README](https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/cd543a149df89434d8a54582c0151c0b945c3d20/README.md). |
| Kim (2014) | Yoon Kim, EMNLP, 1746–1751, DOI 10.3115/v1/D14-1181 일치. [ACL 원문 페이지](https://aclanthology.org/D14-1181/). 구현은 논문에서 영감을 받은 별도 모델이며 원 논문 성능을 재현했다고 주장하지 않음. |
| Hochreiter & Schmidhuber (1997) | 두 저자, Neural Computation 9(8), 1735–1780, DOI 10.1162/neco.1997.9.8.1735 일치. DOI 웹 페이지는 일부 도구에서 403 응답이어서 [Crossref 등록 메타데이터](https://api.crossref.org/works/10.1162/neco.1997.9.8.1735)와 [저자 제공 논문](https://people.idsia.ch/~juergen/lstm1997-2024head.pdf)으로 교차 확인. 접근 차단을 허위 DOI로 해석하지 않음. |
| PyTorch | 설치 환경에 맞는 2.6 문서 확인. [Conv1d](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Conv1d.html), [LSTM](https://docs.pytorch.org/docs/2.6/generated/torch.nn.LSTM.html), [pack_padded_sequence](https://docs.pytorch.org/docs/2.6/generated/torch.nn.utils.rnn.pack_padded_sequence.html). 후자의 두 링크를 참고문헌에 추가. HTTP 200과 API 이름을 직접 확인. |
| Self-Instruct | [ACL 2023 원문](https://aclanthology.org/2023.acl-long.754/)의 저자 7명·제목·13484–13508·DOI를 확인하고 보고서에 추가. |
| Vicuna | [공식 LMSYS 글](https://lmsys.org/blog/2023-03-30-vicuna/)의 제목·2023-03-30·공식 인용을 확인. 제목의 `90%*` 별표를 복구. 해당 수치는 이 프로젝트 정확도가 아니며 참고문헌 제목에만 사용. |
| Koala | [원본 평가 데이터 저장소](https://github.com/arnav-gudibande/koala-test-set)를 확인. 논문인 것처럼 학술지·DOI·연도를 만들지 않음. 학습 사용을 피하라는 원문 요청은 아래 별도 기록. |
| HH-RLHF | [Anthropic 데이터 카드](https://huggingface.co/datasets/Anthropic/hh-rlhf)와 [공식 저장소](https://github.com/anthropics/hh-rlhf)를 확인. 원본 선호도 레이블을 분류 레이블로 사용했다고 주장하지 않음. |
| 라이선스 | AlpacaEval 고정 README의 데이터 CC-BY-NC-4.0 / 코드 Apache-2.0 구분을 확인. [CC 공식 설명](https://creativecommons.org/licenses/by-nc/4.0/)과 포함된 고지 확인. 프로젝트 코드 라이선스가 데이터 이용 조건을 덮어쓴다고 주장하지 않음. |

## 최종 재실험 및 구현 확인

- Koala 원문 요청에 따라 원본 Koala 질문 156개를 모든 분할에서 제외했습니다. OASST 188개도 제외합니다. 현재 데이터는 질문 448개, 응답 1,344개, 템플릿 그룹 429개이며 모델별 448개입니다.
- 고정 원본 GPT·Claude·Gemini 파일의 SHA-256 및 각 805개 항목을 확인했습니다. 최종 응답은 원본에 공통 identity 마스킹을 적용한 내용입니다.
- 기존 수치를 수정해 붙이지 않고 필수 18개 신경망 학습을 포함한 38개 실험을 모두 재실행했습니다. 세 main seed는 42/43/44, split seed는 2026입니다.
- 최종 main 질문 분할은 학습/검증/테스트 278/83/87개이며 테스트 응답은 261개입니다. 모든 입력 조건에 공통입니다.
- CNN 질문 추가 변화는 +2.55%p, paired 95% 구간 [-0.51, +5.69]%p입니다. 해석은 실제 구간의 0 포함 여부에 맞춰 자동 생성합니다.
- 저장 예측에서 38개 실험의 정확도·Macro-F1·혼동행렬을 재계산합니다. 14개 체크포인트는 대표 행뿐 아니라 전체 테스트 예측 확률도 다시 검증합니다. 결과는 `artifacts/verification.json`에 기록합니다.
- Lessons를 출처 검토와 재학습, 대조군, 입력 예산·패딩 테스트, 불확실성, 산출물 검증의 5개 구체적 사례로 보강했습니다. 학생 개인의 경험이나 역할을 만들어 주장하지 않습니다.

## 데이터 이용 범위 수정

Koala의 학습 사용 회피 요청을 instruction tuning과 다른 목적이라는 이유로 해소됐다고 판단하지 않았습니다. 최종 프로토콜은 Koala를 학습·검증·테스트 모두에서 제외하며, 제출 데이터에 Koala나 OASST가 나타나면 검증이 실패합니다. 출처와 제외 근거는 `data/NOTICE.md`, `data/source_manifest.json`, `data/processed/audit.json`에 남깁니다. 제외된 실제 내용과 이전 후보 결과는 제출 패키지에 포함하지 않습니다.

사용 모델은 과거 버전이고 실제 생성 조건이 완전히 동일하지 않습니다. 이번 검증은 만점 보장이 아니라 제출물의 일관성 확인입니다. GitHub 업로드와 수업 제출 채널에 URL을 제출하는 것은 별개입니다.
