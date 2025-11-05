---
marp: true
size: 16:9
paginate: true
transition: fade
style: |
  /* Mark exactly ONE element per slide with the same view-transition-name */
  .diagram {
    /* This name must be unique-per-element but identical across the two slides */
    view-transition-name: diagram;
    display: block;
    margin: 0 auto;
    /* Optional: let the browser do a subtle zoom/position ease */
    contain: layout paint; /* helps perf */
  }

  /* Optional: a couple of sizes/positions to show the morph clearly */
  .w-70 { width: 70%; }
  .w-45 { width: 45%; }
  .mt-0 { margin-top: 0; }
  .mt-6 { margin-top: 6vh; }
---

# Pipeline overview (A)

<!-- Only one element on this page should use the name `diagram` -->
<img class="diagram w-70 mt-0" src="overview1.png" alt="Overview A">
dasdsadas

---

# Pipeline overview (B) — morphs on slide change

<!-- Same name `diagram`, different size/position/src is fine -->
<img class="diagram w-70 mt-0" src="overview2.png" alt="Overview B">
dasdsadas