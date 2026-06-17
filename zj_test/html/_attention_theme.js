(() => {
  if (window.__attentionThemeCanvasPatch) return;
  window.__attentionThemeCanvasPatch = true;

  const fill = new Map([
    ["#08111f", "rgba(255,255,255,.82)"],
    ["#091322", "rgba(255,255,255,.82)"],
    ["#0c0e14", "rgba(255,255,255,.82)"],
    ["#0f1117", "rgba(255,255,255,.82)"],
    ["#101d31", "rgba(255,255,255,.82)"],
    ["#12141c", "rgba(255,255,255,.82)"],
    ["#1a1d27", "rgba(255,255,255,.82)"],
    ["#e4e6ed", "#172033"],
    ["#cfe1ff", "#172033"],
    ["#8b8fa3", "#687083"],
    ["#5a5f73", "#687083"],
    ["#5a6378", "#687083"]
  ]);

  const stroke = new Map([
    ["#08111f", "#dedbd2"],
    ["#091322", "#dedbd2"],
    ["#0c0e14", "#dedbd2"],
    ["#0f1117", "#dedbd2"],
    ["#101d31", "#dedbd2"],
    ["#12141c", "#dedbd2"],
    ["#1a1d27", "#dedbd2"],
    ["#222533", "#dedbd2"],
    ["#243551", "#dedbd2"],
    ["#2a2d3a", "#dedbd2"],
    ["#3a3d4a", "#dedbd2"],
    ["#5a3a4a", "#e4c8bc"],
    ["#5a6378", "#9aa2b4"]
  ]);

  const norm = value => typeof value === "string" ? value.trim().toLowerCase() : value;

  const patch = (prop, map) => {
    let proto = CanvasRenderingContext2D.prototype;
    let desc;
    while (proto && !desc) {
      desc = Object.getOwnPropertyDescriptor(proto, prop);
      proto = Object.getPrototypeOf(proto);
    }
    if (!desc || !desc.set || !desc.get) return;
    Object.defineProperty(CanvasRenderingContext2D.prototype, prop, {
      get() { return desc.get.call(this); },
      set(value) { desc.set.call(this, map.get(norm(value)) || value); }
    });
  };

  patch("fillStyle", fill);
  patch("strokeStyle", stroke);
})();
