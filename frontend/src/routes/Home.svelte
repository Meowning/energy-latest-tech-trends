<script>
  let email = '';
  const submitEmail = () => {
    alert(`입력된 이메일: ${email}`);
  };
</script>

<!-- 스타일 불러오기 -->
<style>
  @import '../assets/styles/home.css';
</style>

<section class="hero">
  <h1>에너지 기술 동향 알리미</h1>
  <p>최신 에너지 기술 뉴스와 트렌드를 한눈에 확인하세요.</p>
</section>

<section class="features">
  <div class="feature-box">
    <h2>📰 실시간 뉴스 크롤링</h2>
    <p>에너지 산업 관련 정보를 매일 업데이트합니다.</p>
  </div>
  <div class="feature-box">
    <h2>📈 기술 동향 분석</h2>
    <p>기사 흐름과 트렌드를 그래프로 시각화합니다.</p>
  </div>
  <div class="feature-box">
    <h2>📬 뉴스레터 구독</h2>
    <p>이메일을 등록하면 최신 동향을 매주 받아보실 수 있어요!</p>
  </div>
</section>

<section class="newsletter">
  <h2>뉴스레터 구독하기</h2>
  <form on:submit|preventDefault={submitEmail}>
    <input
      type="email"
      placeholder="이메일을 입력하세요"
      bind:value={email}
      required
    />
    <button type="submit">구독하기</button>
  </form>
</section>
