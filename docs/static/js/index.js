// Subset of the Academic Project Page Template's index.js (no carousel/jQuery needed).
function copyBibTeX() {
  const code = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const label = button.querySelector('.copy-text');
  const done = function () {
    button.classList.add('copied');
    label.textContent = 'Cop';
    setTimeout(function () { button.classList.remove('copied'); label.textContent = 'Copy'; }, 2000);
  };
  navigator.clipboard.writeText(code.textContent).then(done).catch(function () {
    const ta = document.createElement('textarea');
    ta.value = code.textContent;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    done();
  });
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function () {
  document.querySelector('.scroll-to-top').classList.toggle('visible', window.pageYOffset > 300);
});
