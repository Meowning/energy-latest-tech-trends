<!-- src/routes/Home.svelte -->
<script>
  import { onMount, tick } from 'svelte';
  import '../assets/styles/home.css';

  let email = '';
  function submitEmail() {
    alert(`입력된 이메일: ${email}`);
  }

  onMount(async () => {
    await tick();
    const sections = document.querySelectorAll('.snap-section');
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const visible = entry.intersectionRatio >= 0.5;
        entry.target.querySelector('.content')
             .classList.toggle('visible', visible);
        const action = entry.target.querySelector('.main__action');
        if (action) action.classList.toggle('visible', visible);
      });
    }, { threshold: [0.5] });
    sections.forEach(sec => obs.observe(sec));
  });
</script>

<main>
  <section id="section1" class="snap-section">
    <div class="content">
      <h1>⚡ 에너지 최신기술동향 알리미 📡</h1>
      <p>최신 에너지 기술 뉴스와 트렌드를 한눈에 확인하세요.</p>
    </div>
    <div class="main__action">
      <a class="main__scroll" href="#section2">
        <div class="main__scroll-box">
          <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M0 0h24v24H0z" fill="none"/>
            <path d="M11.9997 13.1716L7.04996 8.22186L5.63574 9.63607L11.9997 16L18.3637 9.63607L16.9495 8.22186L11.9997 13.1716Z" fill="rgba(28,28,30,1)"/>
          </svg>
        </div>
        <span class="main__scroll-text">Scroll</span>
      </a>
    </div>
  </section>

  <section id="section2" class="snap-section">
    <div class="content">
      <h2>📰 실시간 간행물/보도자료 크롤링</h2>
      <p>공공기관에서 발행하는 에너지 산업 관련 정보를 매일 업데이트합니다.</p>
    </div>
  </section>

  <section class="snap-section">
    <div class="content">
      <h2>📄 콘텐츠 자동 정리</h2>
      <p>수집된 자료를 제목, 요약, 발행일, 표지 사진, PDF 파일, 원문 링크 형태로 깔끔하게 정리해드립니다.</p>
    </div>
  </section>

    <section class="snap-section">
    <div class="content">
      <h2>🕘 운영 시간</h2>
      <h3 >월요일 ~ 금요일, 09:00 ~ 18:00</h3>
      <p>⛔️ 그 외 시간 및 주말은 알리미가 퇴근합니다.</p>
    </div>
  </section>

  <section class="snap-section newsletter">
    <div class="content">
      <h2>메일로 최신 소식을 빠르게 받아보세요.</h2>
      <form on:submit|preventDefault={submitEmail} class="newsletter">
        <div class="input-wrapper">
          <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <g data-name="Layer 2">
              <g data-name="inbox">
                <rect width="24" height="24" transform="rotate(180 12 12)" opacity="0"></rect>
                <path d="M20.79 11.34l-3.34-6.68A3 3 0 0 0 14.76 3H9.24a3 3 0 0 0-2.69 1.66l-3.34 6.68a2 2 0 0 0-.21.9V18a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-5.76a2 2 0 0 0-.21-.9zM8.34 5.55a1 1 0 0 1 .9-.55h5.52a1 1 0 0 1 .9.55L18.38 11H16a1 1 0 0 0-1 1v2a1 1 0 0 1-1 1h-4a1 1 0 0 1-1-1v-2a1 1 0 0 0-1-1H5.62z"/>
              </g>
            </g>
          </svg>
          <input
            type="email"
            name="email"
            bind:value={email}
            class="input"
            placeholder="info@gmail.com"
          />
          <button class="Subscribe-btn" type="submit">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            class="arrow"
          >
          <path
            d="M5 12h14M12 5l7 7-7 7"
            stroke="currentColor"
            stroke-width="2.5"
            fill="none"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
          </svg>
            등록
          </button>
        </div>
      </form>
    </div>
  </section>
</main>
