(function () {
  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }
  ready(function () {
    setupCites();
    setupSee();
    setupStack();
    setupBib();
    setupToc();
    setupViews();
  });
})();

function setupViews() {
  var nodes = document.querySelectorAll("[data-views]");
  if (!nodes.length) return;
  var sessionKey = "diffusion-is-matching-viewed";
  var shouldHit = false;
  nodes.forEach(function (el) {
    if (el.getAttribute("data-views") === "hit") shouldHit = true;
  });
  var endpoint = shouldHit && !sessionStorage.getItem(sessionKey) ? "hit" : "get";
  fetch("https://abacus.jasoncameron.dev/" + endpoint + "/waynejin0918/diffusion-is-matching")
    .then(function (res) { return res.json(); })
    .then(function (data) {
      if (typeof data.value !== "number") return;
      if (endpoint === "hit") sessionStorage.setItem(sessionKey, "1");
      var text = String(data.value).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
      nodes.forEach(function (el) {
        var label = el.getAttribute("data-views-label");
        el.textContent = label ? text + " " + label : text;
      });
    })
    .catch(function () {});
}

function setupCites() {
  const tip = document.createElement("div");
  tip.className = "cite-tip";
  document.body.appendChild(tip);

  function hide() {
    tip.classList.remove("on");
  }

  document.querySelectorAll("a.cite").forEach(function (a) {
    const href = a.getAttribute("href") || "";
    const id = href.startsWith("#ref-")
      ? href
      : a.textContent.trim()
        ? "#ref-" + a.textContent.trim()
        : "";
    const li = id ? document.querySelector(id) : null;
    const text = li ? li.textContent.replace(/\s+/g, " ").trim() : "";

    a.addEventListener("mouseenter", function (ev) {
      if (!text) return;
      tip.textContent = text;
      tip.classList.add("on");
      const r = ev.target.getBoundingClientRect();
      const left = Math.min(r.left, window.innerWidth - 360);
      tip.style.left = Math.max(12, left) + "px";
      tip.style.top = r.bottom + 8 + "px";
    });
    a.addEventListener("mouseleave", hide);

    a.addEventListener("click", function (ev) {
      if (!li) return;
      ev.preventDefault();
      document.querySelectorAll(".refs li.flash").forEach(function (n) {
        n.classList.remove("flash");
      });
      li.classList.add("flash");
      li.scrollIntoView({ behavior: "smooth", block: "center" });
      history.replaceState(null, "", id);
      setTimeout(function () {
        li.classList.remove("flash");
      }, 1600);
    });
  });
}

function setupSee() {
  document.querySelectorAll(".fig-see").forEach(function (fig) {
    const buttons = fig.querySelectorAll(".see-toggle button");
    const note = fig.querySelector(".see-note");
    const notes = {
      bi: fig.dataset.noteBi || "",
      ar: fig.dataset.noteAr || "",
    };
    function setMode(mode) {
      fig.dataset.mode = mode;
      buttons.forEach(function (b) {
        b.classList.toggle("on", b.dataset.mode === mode);
      });
      if (note) note.textContent = notes[mode];
    }
    buttons.forEach(function (b) {
      b.addEventListener("click", function () {
        setMode(b.dataset.mode);
      });
    });
    setMode(fig.dataset.mode || "bi");
  });
}

function setupBib() {
  document.querySelectorAll(".bib-copy").forEach(function (btn) {
    const box = btn.closest(".bib");
    const code = box ? box.querySelector("code") : null;
    if (!code) return;
    const done = btn.textContent;
    btn.addEventListener("click", function () {
      navigator.clipboard.writeText(code.textContent).then(function () {
        btn.textContent = "✓";
        setTimeout(function () {
          btn.textContent = done;
        }, 1200);
      });
    });
  });
}

function setupToc() {
  const links = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
  if (!links.length) return;
  const secs = links
    .map(function (a) {
      return { a: a, el: document.querySelector(a.getAttribute("href")) };
    })
    .filter(function (s) {
      return s.el;
    });
  function sync() {
    var cur = secs[0];
    secs.forEach(function (s) {
      if (s.el.getBoundingClientRect().top <= 140) cur = s;
    });
    links.forEach(function (a) {
      a.classList.toggle("on", cur && a === cur.a);
    });
  }
  document.addEventListener("scroll", sync, { passive: true });
  sync();
}

function setupStack() {
  document.querySelectorAll(".fig-stack").forEach(function (fig) {
    const items = fig.querySelectorAll(".stack li");
    items.forEach(function (li) {
      li.addEventListener("click", function () {
        items.forEach(function (n) {
          n.classList.toggle("on", n === li);
        });
      });
    });
  });
}
