// Small progressive-enhancement script:
// - Active nav highlighting based on scroll position
// - Reveal-on-scroll for section cards

(function () {
  'use strict';

  // --- Active nav link ---
  const navLinks = document.querySelectorAll('.nav-links a');
  const sections = Array.from(navLinks)
    .map(a => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);

  function setActive() {
    const y = window.scrollY + 120;
    let current = sections[0];
    for (const s of sections) {
      if (s.offsetTop <= y) current = s;
    }
    navLinks.forEach(a => a.classList.remove('active'));
    const link = document.querySelector(`.nav-links a[href="#${current.id}"]`);
    if (link) link.classList.add('active');
  }

  window.addEventListener('scroll', setActive, { passive: true });
  setActive();

  // --- Reveal on scroll ---
  const revealSelectors = [
    '.metric',
    '.stack-card',
    '.download-card',
    '.note-card',
    '.demo-gallery figure',
    '.results-figs figure'
  ];
  const revealTargets = document.querySelectorAll(revealSelectors.join(','));

  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -60px 0px' });

    revealTargets.forEach(el => {
      el.classList.add('reveal');
      io.observe(el);
    });
  }
})();
