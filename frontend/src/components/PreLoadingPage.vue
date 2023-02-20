<template>
  <div class="preloading-page">
    <div class="preloading-content">
      <img class="animating-image" :src="imageUrl" />
      <img class="preloading-image" src="../assets/images/maintext.jpg" />
      <!-- <p class="preloading-text">로딩중...</p> -->
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      images: [
        require("../assets/images/icon1.png"),
        require("../assets/images/icon2.png"),
        require("../assets/images/icon3.png"),
        require("../assets/images/icon4.png"),
        require("../assets/images/icon5.png"),
      ],
      imageIndex: 0,
    };
  },
  setup() {
    const setScreenSize = function () {
      let vh = window.innerHeight * 0.01;
      document.documentElement.style.setProperty("--vh", `${vh}px`); //핸드폰 화면에 맞게 vh 새로 설정
    };
    setScreenSize();
    window.addEventListener("resize", () => setScreenSize());
    setInterval(() => {});
  },
  computed: {
    imageUrl() {
      return this.images[this.imageIndex % this.images.length];
    },
  },
  mounted() {
    setInterval(() => {
      this.imageIndex++;
    }, 400);
  },
};
</script>

<style>
.preloading-page {
  height: calc(var(--vh, 1vh) * 100);
  background: #fff;
  overflow: hidden;
  background-repeat: no-repeat;
  background-attachment: fixed;
  background-size: cover;
  -webkit-background-size: cover;
  -moz-background-size: cover;
  -o-background-size: cover;
}
.preloading-content {
  font-family: korFont2;
  font-size: 30px;
  position: absolute;
  margin-left: auto;
  margin-right: auto;
  left: 0;
  right: 0;
  top: 55%;
  text-align: center;
  background: none;
  color: inherit;
  border: none;
  padding: 0;
  cursor: pointer;
  outline: inherit;
  color: #000;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.preloading-text {
  color: #3b3b3bb9;
  margin-top: 20px;
  font-size: 17px;
  animation: blinker 1.5s linear infinite;
  margin-left: 10px;
}
.preloading-image {
  width: 100px;
}
.animating-image {
  width: 100px;
  position: absolute;
  top: -100px;
}

@keyframes blinker {
  50% {
    opacity: 0;
  }
}
</style>
