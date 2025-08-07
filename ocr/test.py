from hanspell import spell_checker

# 안전한 맞춤법 검사 함수
def safe_spell_check(sent: str) -> str:
    try:
        # Hanspell check 실행
        res = spell_checker.check(sent)
        # 디버그용: 내부 구조 확인
        data = res._asdict() if hasattr(res, '_asdict') else res.__dict__ if hasattr(res, '__dict__') else {}
        print("----- Debug hanspell result data -----")
        print(data)
        # 'checked' 필드 우선 사용, 없으면 'original', 없으면 원본
        return data.get('checked') or data.get('original') or sent
    except Exception as e:
        print(f"맞춤법 검사 예외 발생: {e}")
        return sent

if __name__ == '__main__':
    samples = [
        "테스트 문장입니다.",
        "이것은 아주 길어서 에러가 날 수도 있는 문장입니다." * 20,
        "짧음"
    ]
    for s in samples:
        print(f"\nOriginal: {s}")
        corrected = safe_spell_check(s)
        print(f"Corrected: {corrected}\n")
