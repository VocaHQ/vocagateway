document.documentElement.classList.add("js");

const menuToggle = document.querySelector("[data-menu-toggle]");
const navigation = document.querySelector("[data-navigation]");
const menuLabel = menuToggle?.querySelector(".sr-only");

if (menuToggle && navigation) {
  const closeNavigation = ({ returnFocus = false } = {}) => {
    menuToggle.setAttribute("aria-expanded", "false");
    if (menuLabel) menuLabel.textContent = "Open navigation";
    navigation.classList.remove("is-open");
    if (returnFocus) menuToggle.focus();
  };

  menuToggle.addEventListener("click", () => {
    const isOpen = menuToggle.getAttribute("aria-expanded") === "true";
    if (isOpen) {
      closeNavigation();
    } else {
      menuToggle.setAttribute("aria-expanded", "true");
      if (menuLabel) menuLabel.textContent = "Close navigation";
      navigation.classList.add("is-open");
    }
  });

  navigation.addEventListener("click", (event) => {
    if (event.target instanceof HTMLAnchorElement) {
      closeNavigation();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && menuToggle.getAttribute("aria-expanded") === "true") {
      closeNavigation({ returnFocus: true });
    }
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth > 920 && menuToggle.getAttribute("aria-expanded") === "true") {
      closeNavigation();
    }
  });
}

const timeNode = document.querySelector("[data-local-time]");
const yearNodes = document.querySelectorAll("[data-current-year]");

function updateClock() {
  const now = new Date();
  if (timeNode) {
    timeNode.textContent = new Intl.DateTimeFormat(undefined, {
      hour: "numeric",
      minute: "2-digit",
    }).format(now);
  }
  yearNodes.forEach((node) => {
    node.textContent = String(now.getFullYear());
  });
}

updateClock();
window.setInterval(updateClock, 30_000);

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const revealNodes = document.querySelectorAll(".reveal");

if (reduceMotion.matches || !("IntersectionObserver" in window)) {
  revealNodes.forEach((node) => node.classList.add("is-visible"));
} else {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.14 },
  );

  revealNodes.forEach((node) => revealObserver.observe(node));

  window.setTimeout(() => {
    revealNodes.forEach((node) => node.classList.add("is-visible"));
  }, 900);
}

const faqItems = document.querySelectorAll(".faq-list details");
faqItems.forEach((detail) => {
  detail.addEventListener("toggle", () => {
    if (!detail.open) return;
    faqItems.forEach((other) => {
      if (other !== detail) other.open = false;
    });
  });
});
