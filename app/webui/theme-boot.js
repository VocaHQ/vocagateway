/* FOUC-safe theme boot: runs before body paint. Preference is system unless
   localStorage.vocaphone.theme is light/dark/system. */
(function () {
  "use strict";
  try {
    var key = "vocaphone.theme";
    var pref = localStorage.getItem(key);
    if (pref !== "light" && pref !== "dark" && pref !== "system") pref = "system";
    var systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    var resolved = pref === "light" || pref === "dark" ? pref : systemDark ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", resolved);
    document.documentElement.setAttribute("data-theme-preference", pref);
    var meta = document.getElementById("meta-theme-color");
    if (meta) meta.setAttribute("content", resolved === "dark" ? "#141614" : "#f7f6f3");
  } catch (e) {
    document.documentElement.setAttribute("data-theme", "light");
    document.documentElement.setAttribute("data-theme-preference", "system");
  }
})();
