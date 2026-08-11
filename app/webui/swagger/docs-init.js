window.addEventListener("DOMContentLoaded", function () {
  window.ui = SwaggerUIBundle({
    url: "/openapi.json",
    dom_id: "#swagger-ui",
    layout: "BaseLayout",
    deepLinking: true,
    showExtensions: true,
    showCommonExtensions: true,
    // Keeps the token entered in Authorize across reloads, so paging through
    // the admin endpoints does not mean re-pasting it every time. It lands in
    // this browser's localStorage; use a private window on a shared machine.
    persistAuthorization: true,
    // Only the apis preset: this vendored bundle has no standalone preset
    // (that ships as a separate swagger-ui-standalone-preset.js), and
    // BaseLayout does not need it.
    presets: [SwaggerUIBundle.presets.apis],
  });
});
