<template>
  <desktop-scene v-if="desktop"></desktop-scene>
  <PreLoadingPage v-if="mobile"></PreLoadingPage>
  <MainPage
    @ToFirstScene="ToFirstScene"
    @toggleSound="toggleSound"
    v-bind:isPlaying="isPlaying"
    v-if="showMain"
  ></MainPage>

  <FirstScene
    @ToSecondScene="ToSecondScene"
    @toggleSound="toggleSound"
    v-bind:isPlaying="isPlaying"
    v-if="showFirst"
  ></FirstScene>

  <SecondScene
    @toggleSound="toggleSound"
    v-bind:isPlaying="isPlaying"
    v-if="showSecond"
  ></SecondScene>
</template>

<script>
import DesktopScene from "./components/DesktopScene.vue";
import PreLoadingPage from "./components/PreLoadingPage.vue";
import MainPage from "./views/Main/MainPage.vue";
import FirstScene from "./views/First/FirstScene.vue";
import SecondScene from "./views/Second/SecondScene.vue";

export default {
  name: "App",
  components: {
    DesktopScene,
    MainPage,
    FirstScene,
    SecondScene,
    PreLoadingPage,
  },
  data() {
    return {
      mobile: false,
      desktop: false,
      showMain: false,
      showFirst: false,
      showSecond: false,
      isPlaying: true,
      device: "",
    };
  },
  methods: {
    ToFirstScene() {
      this.showMain = false;
      this.showFirst = true;
    },
    ToSecondScene() {
      this.showFirst = false;
      this.showSecond = true;
    },

    leave(event) {
      event.preventDefault();
      event.returnValue = "";
    },
    checkMobile() {
      if (
        /iP(hone|od|ad)|Android|BlackBerry|IEMobile|NetFront|Silk-Accelerated|(hpw|web)OS|Fennec|Minimo|Opera M(obi|ini)|Blazer|Dolfin|Dolphin|Skyfire|Zune|Lumia/g.test(
          navigator.userAgent
        )
      ) {
        this.device = "mobile";
      } else {
        this.device = "etc";
      }
      console.log(this.device);
    },
    toggleSound() {
      this.isPlaying = !this.isPlaying;
    },
  },
  created() {
    this.checkMobile();
    if (this.device === "mobile") {
      this.mobile = true;
      this.desktop = false;
    } else {
      this.mobile = false;
      this.desktop = true;
    }
  },
  mounted() {
    window.addEventListener("beforeunload", this.leave);
    //나중에 아래 코드를 로딩이 완료 된 후에 실행되는 것으로 변환
    if (this.device === "mobile") {
      setTimeout(() => {
        if (document.readyState === "complete") {
          this.mobile = false;
          this.showMain = true;
        } else {
          window.addEventListener("load", () => {
            this.mobile = false;
            this.showMain = true;
          });
        }
      }, 1000);

      // window.onload = () => {
      //   this.mobile = false;
      //   this.showMain = true;
      // };
    } //모바일일때만 해당 코드 실행
  },

  beforeUnmount() {
    window.removeEventListener("beforeunload", this.leave);
  },
};
</script>
<style>
html,
body,
div,
span,
applet,
object,
iframe,
h1,
h2,
h3,
h4,
h5,
h6,
p,
blockquote,
pre,
a,
abbr,
acronym,
address,
big,
cite,
code,
del,
dfn,
em,
img,
ins,
kbd,
q,
s,
samp,
small,
strike,
strong,
sub,
sup,
tt,
var,
b,
u,
i,
center,
dl,
dt,
dd,
ol,
ul,
li,
fieldset,
form,
label,
legend,
table,
caption,
tbody,
tfoot,
thead,
tr,
th,
td,
article,
aside,
canvas,
details,
embed,
figure,
figcaption,
footer,
header,
hgroup,
menu,
nav,
output,
ruby,
section,
summary,
time,
mark,
audio,
video {
  margin: 0;
  padding: 0;
  border: 0;
  font-size: 100%;
  font: inherit;
  vertical-align: baseline;
}
/* HTML5 display-role reset for older browsers */
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
menu,
nav,
section {
  display: block;
}
body {
  line-height: 1;
}
ol,
ul {
  list-style: none;
}
blockquote,
q {
  quotes: none;
}
blockquote:before,
blockquote:after,
q:before,
q:after {
  content: "";
  content: none;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #fff;
  background-color: #151515;
  position: relative;
}

body,
html {
  margin: 0;
  padding: 0;
}
* {
  box-sizing: border-box;
}

@font-face {
  font-family: korFont3;
  src: url(./assets/fonts/human.ttf);
}
@font-face {
  font-family: korFont1;
  src: url(./assets/fonts/uhbee.ttf);
}
@font-face {
  font-family: korFont1Bold;
  src: url(./assets/fonts/uhbeebold.ttf);
}
@font-face {
  font-family: korFont2;
  src: url(./assets/fonts/micegothic.ttf);
}

.page-enter-active,
.page-leave-active {
  transition: opacity 0.5s;
}
.page-enter,
.page-leave-to {
  opacity: 0;
}
</style>
