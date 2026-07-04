(() => {
  const slides = Array.from(document.querySelectorAll('.slide'));
  const total = slides.length;
  const fill = document.getElementById('progressFill');
  const counter = document.getElementById('counter');
  const dotsWrap = document.getElementById('dots');
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');

  let current = 0;
  let locked = false;

  // build dots
  slides.forEach((s, i) => {
    const d = document.createElement('button');
    d.className = 'dot';
    d.setAttribute('aria-label', 'Go to slide ' + (i + 1));
    d.addEventListener('click', () => go(i));
    dotsWrap.appendChild(d);
  });
  const dots = Array.from(dotsWrap.children);

  document.querySelector('.counter-total').textContent = '/ ' + String(total).padStart(2, '0');

  function render() {
    slides.forEach((s, i) => s.classList.toggle('is-active', i === current));
    dots.forEach((d, i) => d.classList.toggle('active', i === current));
    fill.style.width = (total === 1 ? 100 : (current / (total - 1)) * 100) + '%';
    counter.textContent = String(current + 1).padStart(2, '0');
    prevBtn.disabled = current === 0;
    nextBtn.disabled = current === total - 1;
    document.body.classList.toggle('on-dark', slides[current].matches('.closing, .nb'));
  }

  function go(i) {
    if (locked) return;
    const next = Math.max(0, Math.min(total - 1, i));
    if (next === current) return;
    current = next;
    locked = true;
    render();
    setTimeout(() => (locked = false), 420);
  }

  const nextSlide = () => go(current + 1);
  const prevSlide = () => go(current - 1);

  prevBtn.addEventListener('click', prevSlide);
  nextBtn.addEventListener('click', nextSlide);
  document.querySelectorAll('[data-next]').forEach(b => b.addEventListener('click', nextSlide));
  document.querySelectorAll('[data-goto]').forEach(b =>
    b.addEventListener('click', () => go(parseInt(b.dataset.goto, 10)))
  );

  // keyboard
  window.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') { e.preventDefault(); nextSlide(); }
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); prevSlide(); }
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(total - 1);
  });

  // wheel (debounced, horizontal-ish)
  let wheelLock = false;
  window.addEventListener('wheel', e => {
    if (Math.abs(e.deltaY) < 24 && Math.abs(e.deltaX) < 24) return;
    if (wheelLock) return;
    wheelLock = true;
    (e.deltaY > 0 || e.deltaX > 0) ? nextSlide() : prevSlide();
    setTimeout(() => (wheelLock = false), 650);
  }, { passive: true });

  // touch swipe
  let tx = 0, ty = 0;
  window.addEventListener('touchstart', e => { tx = e.changedTouches[0].clientX; ty = e.changedTouches[0].clientY; }, { passive: true });
  window.addEventListener('touchend', e => {
    const dx = e.changedTouches[0].clientX - tx;
    const dy = e.changedTouches[0].clientY - ty;
    if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy)) dx < 0 ? nextSlide() : prevSlide();
  }, { passive: true });

  render();
})();
