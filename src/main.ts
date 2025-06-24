import "./style.css";
import { Fn, select, vec2, shapeCircle, instanceIndex } from "three/tsl";
import * as THREE from "three/webgpu";
import { FlockSystem } from "./flock-system";
import { Scene } from "./scene";
import { GUI } from "three/addons/libs/lil-gui.module.min.js";

const gui = new GUI();
const mainScene = new Scene();
mainScene.setup();

const renderer = mainScene.renderer;
const controls = mainScene.controls;

const flockSystem = new FlockSystem();

const globalConfig = {
  renderGrid: false,
};
const gridParameters = gui.addFolder('Grid Parameters')
gridParameters.add(flockSystem, "gridSize", 100, 2000, 100).name("Grid size");
gridParameters.add(flockSystem, "cellSize", 4, 64, 2).name("Cell size");
gridParameters.add(flockSystem, "maxBoidsPerCell", 1, 200, 5).name("Max boids per cell");
gridParameters.add(globalConfig, "renderGrid").name("Render grid");
const boidParameters = gui.addFolder("Boid parameters");
boidParameters
  .add(flockSystem, "boidCount", 10000, 400000, 10000)
  .name("Boid count");
boidParameters
  .add(flockSystem, "separationForce", 0, 2, 0.01)
  .name("Separation force");
boidParameters
  .add(flockSystem, "alignmentForce", 0, 2, 0.01)
  .name("Alignment force");
boidParameters
  .add(flockSystem, "cohesionForce", 0, 2, 0.01)
  .name("Cohesion force");
boidParameters.add(flockSystem, "maxSpeed", 0, 1, 0.01).name("Max speed");
boidParameters.add(flockSystem, "maxForce", 0, 1, 0.01).name("Max force");

const start = async () => {
  flockSystem.reset();
  mainScene.scene.clear();

  flockSystem.setupMouse(mainScene.camera);
  const computeInitializeBoids = flockSystem.initializeBoids();
  const clearGrid = flockSystem.clearGrid();
  const insertBoidsIntoGrid = flockSystem.insertBoidsIntoGrid();
  const computeUpdatePosition = flockSystem.computeUpdatePosition();
  const computeFlockPosition = flockSystem.computeFlockPosition();

  await renderer.computeAsync(computeInitializeBoids);
  const updateLoop = async () => {
    await renderer.computeAsync(clearGrid);
    await renderer.computeAsync(insertBoidsIntoGrid);
    await renderer.computeAsync(computeUpdatePosition);
    await renderer.computeAsync(computeFlockPosition);
  };

  const renderSetup = () => {
    const shapeBox = Fn(() => {
      const s = flockSystem.uniforms.boid.size;
      return vec2(s, s);
    });
    const scaleNode = mainScene.cameraPosition
      .distance(controls.target)
      .mul(0.001)
      .add(flockSystem.uniforms.boid.size);
    const shapeCircleOrBox = Fn(() => {
      return select(
        mainScene.cameraZoom.lessThan(300),
        shapeCircle(),
        shapeBox()
      );
    });

    const material = new THREE.SpriteNodeMaterial();
    material.colorNode =
      flockSystem.uniforms.boid.colors.element(instanceIndex);
    material.positionNode = flockSystem.uniforms.boid.positions.toAttribute();
    material.scaleNode = scaleNode;
    material.alphaToCoverage = true;
    material.transparent = false;
    material.depthWrite = false;
    material.blending = THREE.AdditiveBlending;
    material.opacityNode = shapeCircleOrBox();

    const particles = new THREE.Sprite(material);
    particles.count = flockSystem.boidCount;
    particles.frustumCulled = false;
    mainScene.scene.add(particles);

    if (globalConfig.renderGrid) {
      const helper = new THREE.GridHelper(
        flockSystem.gridMax * 2,
        flockSystem.gridMax / flockSystem.cellSize,
        0x303030,
        0x303030
      );
      mainScene.scene.add(helper);
    }
  };

  renderSetup();
  mainScene.setAnimationLoop(updateLoop);
};

gui.onChange(start);
start();
