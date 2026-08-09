/* FOUC-safe theme boot: runs before body paint. Preference is system unless
   localStorage.vocagateway.theme is light/dark/system. */
(function () {
  "use strict";
  function setFavicon(resolved) {
    var href = resolved === "dark" ? "/assets/favicon-dark.svg" : "/assets/favicon-light.svg";
    var favicon = document.getElementById("favicon");
    var apple = document.getElementById("apple-touch-icon");
    if (favicon) favicon.setAttribute("href", href);
    if (apple) apple.setAttribute("href", href);
  }
  try {
    var key = "vocagateway.theme";
    var pref = localStorage.getItem(key);
    if (pref !== "light" && pref !== "dark" && pref !== "system") pref = "system";
    var systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    var resolved = pref === "light" || pref === "dark" ? pref : systemDark ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", resolved);
    document.documentElement.setAttribute("data-theme-preference", pref);
    var meta = document.getElementById("meta-theme-color");
    if (meta) meta.setAttribute("content", resolved === "dark" ? "#141614" : "#f7f6f3");
    setFavicon(resolved);
  } catch (e) {
    document.documentElement.setAttribute("data-theme", "light");
    document.documentElement.setAttribute("data-theme-preference", "system");
  }
})();
