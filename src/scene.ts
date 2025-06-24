import Stats from "stats-gl";
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/examples/jsm/Addons.js";
import { uniform } from "three/tsl";

class Scene {
  scene!: THREE.Scene;
  camera!: THREE.PerspectiveCamera;
  stats!: Stats;
  renderer!: THREE.WebGPURenderer;
  controls!: OrbitControls;
  cameraZoom!: THREE.TSL.ShaderNodeObject<THREE.UniformNode<number>>;
  cameraPosition!: THREE.TSL.ShaderNodeObject<THREE.UniformNode<THREE.Vector3>>;
  constructor() {}

  setup() {
    this.setupScene();
    this.setupRenderer();
    this.setupStats();
    this.setupCamera();
    this.setupControls();
    this.cameraPosition = uniform(this.camera.position, "vec3");
    this.cameraZoom = uniform(
      this.camera.position.distanceTo(this.controls.target),
      "float"
    );
  }

  setupScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111111);
  }

  setupCamera() {
    this.camera = new THREE.PerspectiveCamera(
      50,
      window.innerWidth / window.innerHeight,
      0.1,
      10000
    );
    this.camera.position.y = 50;
  }

  setupControls() {
    const controls = new OrbitControls(this.camera, this.renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 5;
    controls.enableRotate = false;
    // controls.maxDistance = 200;
    controls.update();
    this.controls = controls;
  }

  setupStats() {
    const stats = new Stats({
      precision: 3,
      horizontal: false,
      trackGPU: true,
      trackCPT: true,
    });
    document.body.appendChild(stats.dom);
    this.stats = stats;
  }

  setupRenderer() {
    const renderer = new THREE.WebGPURenderer({
      samples: 0,
      antialias: false,
      powerPreference: "high-performance",
    });

    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    this.renderer = renderer;
  }

  setAnimationLoop(callback: () => Promise<void>) {
    this.renderer.setAnimationLoop(async () => {
      await this.animate(callback);
    });
  }

  private async animate(callback: () => Promise<void>) {
    this.stats.update();
    this.controls.update();
    
    this.cameraZoom.value = this.camera.position.distanceTo(
      this.controls.target
    );
    await callback();

    // Render the scene from the perspective of the camera
    await this.renderer.renderAsync(this.scene, this.camera);
  }
}

export { Scene };
