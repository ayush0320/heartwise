/* HeartWise front-end helpers: theme, nav, tabs, gauge, SHAP bars, charts. No build step needed. */
(function () {
  const root = document.documentElement;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));

  /* ---- theme ---- */
  const themeBtn = $('#theme-btn');
  function applyThemeIcon() {
    const dark = root.dataset.theme === 'dark' || (root.dataset.theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
    if (themeBtn) themeBtn.innerHTML = dark ? themeBtn.dataset.sun : themeBtn.dataset.moon;
  }
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const dark = root.dataset.theme === 'dark' || (root.dataset.theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
      root.dataset.theme = dark ? 'light' : 'dark';
      try { localStorage.setItem('hw-theme', root.dataset.theme); } catch (e) {}
      applyThemeIcon();
      window.dispatchEvent(new Event('themechange'));
    });
    applyThemeIcon();
  }
  const menuBtn = $('#menu-btn');
  if (menuBtn) menuBtn.addEventListener('click', () => $('.nav').classList.toggle('open'));

  /* ---- generic behaviours ---- */
  $$('form[data-confirm]').forEach(f => f.addEventListener('submit', e => { if (!confirm(f.dataset.confirm)) e.preventDefault(); }));
  $$('[data-fill]').forEach(b => b.addEventListener('click', () => {
    const vals = JSON.parse(b.dataset.fill);
    Object.keys(vals).forEach(k => { const el = document.getElementById('f-' + k); if (el) { el.value = vals[k]; el.dispatchEvent(new Event('input', { bubbles: true })); } });
  }));
  $$('.tabs').forEach(tabs => {
    const btns = $$('button', tabs);
    btns.forEach(b => b.addEventListener('click', () => {
      btns.forEach(x => x.classList.toggle('on', x === b));
      $$('.tabpane').forEach(p => p.classList.toggle('on', p.id === b.dataset.tab));
      window.dispatchEvent(new Event('resize'));
    }));
  });
  setTimeout(() => $$('.flash.ok').forEach(f => f.style.display = 'none'), 6000);

  /* ---- helpers ---- */
  const css = n => getComputedStyle(root).getPropertyValue(n).trim();
  const NS = 'http://www.w3.org/2000/svg';
  const pct = v => (Math.round(v * 10) / 10).toFixed(1);
  const BAND = { low: 'Low', moderate: 'Moderate', high: 'High' };

  /* ---- gauge: semicircle with low / moderate / high zones ---- */
  function gauge(el, p, edges, threshold) {
    const cx = 150, cy = 150, r = 112;
    const pt = (q, rad = r) => { const a = Math.PI * (1 - q); return [cx + rad * Math.cos(a), cy - rad * Math.sin(a)]; };
    const arc = (q1, q2, col) => { const [x1, y1] = pt(q1), [x2, y2] = pt(q2); return `<path class="arc" d="M${x1} ${y1} A${r} ${r} 0 0 1 ${x2} ${y2}" stroke="${col}"/>`; };
    const [lo, hi] = edges, g = 0.006;
    const [nx, ny] = pt(Math.min(Math.max(p, 0), 1), r - 20), [tx1, ty1] = pt(threshold, r - 16), [tx2, ty2] = pt(threshold, r + 18);
    const band = p < lo ? 'low' : p < hi ? 'moderate' : 'high';
    el.innerHTML = `<svg class="gauge" viewBox="0 0 300 236" role="img" aria-label="Estimated likelihood ${pct(p * 100)} percent, ${BAND[band]} band">
      ${arc(0, lo - g, 'var(--ok)')}${arc(lo + g, hi - g, 'var(--warn)')}${arc(hi + g, 1, 'var(--bad)')}
      <line x1="${tx1}" y1="${ty1}" x2="${tx2}" y2="${ty2}" stroke="var(--ink)" stroke-width="2" stroke-dasharray="3 2"/>
      <line class="needle" x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}"/><circle class="hub" cx="${cx}" cy="${cy}" r="8"/>
      <text class="cap" x="${cx - r}" y="${cy + 22}" text-anchor="middle">0%</text><text class="cap" x="${cx + r}" y="${cy + 22}" text-anchor="middle">100%</text>
      <text class="big" x="${cx}" y="${cy + 62}" text-anchor="middle">${pct(p * 100)}%</text>
      <text class="cap" x="${cx}" y="${cy + 82}" text-anchor="middle">${BAND[band]} estimated likelihood</text></svg>`;
  }

  /* ---- SHAP bars: red raises the estimate, green lowers it ---- */
  function shap(el, rows, opts = {}) {
    const max = opts.max || Math.max(10, ...rows.map(r => Math.abs(r.points)));
    const show = opts.limit ? rows.slice(0, opts.limit) : rows;
    el.innerHTML = show.map(r => {
      const w = Math.min(Math.abs(r.points) / max, 1) * 50, up = r.points > 0;
      const left = up ? 50 : 50 - w;
      return `<div class="shap-row"><div class="lab">${r.label}<small>${r.value_text || ''}</small></div>
        <div class="track"><div class="bar ${up ? 'up' : 'down'}" style="left:${left}%;width:${Math.max(w, .6)}%"></div></div>
        <div class="pts ${up ? 'up' : 'down'}">${(r.points > 0 ? '+' : '') + r.points.toFixed(1)}</div></div>`;
    }).join('');
  }

  /* ---- Chart.js defaults that follow the theme ---- */
  const charts = [];
  function themeCharts() {
    if (!window.Chart) return;
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.color = css('--ink-3');
    Chart.defaults.borderColor = css('--line');
    charts.forEach(c => { try { c.update(); } catch (e) {} });
  }
  function chart(id, cfg) {
    const el = document.getElementById(id);
    if (!el || !window.Chart) return null;
    themeCharts();
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;   // respect the visitor's reduce-motion setting
    cfg.options = Object.assign({ responsive: true, maintainAspectRatio: false, animation: still ? false : { duration: 500 } }, cfg.options || {});
    const c = new Chart(el, cfg);
    charts.push(c);
    return c;
  }
  window.addEventListener('themechange', themeCharts);
  const colors = () => ({ brand: css('--brand'), teal: css('--teal'), ok: css('--ok'), warn: css('--warn'), bad: css('--bad'), info: css('--info'), ink: css('--ink-2') });
  function line(id, labels, datasets, opts = {}) {
    return chart(id, { type: 'line', data: { labels, datasets }, options: Object.assign({ plugins: { legend: { display: datasets.length > 1 } }, scales: { y: { beginAtZero: !!opts.zero } }, elements: { line: { tension: .3 }, point: { radius: 3 } } }, opts.options || {}) });
  }
  const csrf = () => (document.querySelector('meta[name=csrf]') || {}).content || '';
  window.HW = { gauge, shap, chart, line, colors, css, csrf, pct, $, $$, BAND };
})();
