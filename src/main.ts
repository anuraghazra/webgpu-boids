import "./style.css";
import { Fn, select, vec2, shapeCircle, instanceIndex } from "three/tsl";
import * as THREE from "three/webgpu";
import { FlockSystem } from "./flock-system";
import { Scene } from "./scene";

const mainScene = new Scene();
mainScene.setup();

const renderer = mainScene.renderer;
const controls = mainScene.controls;

const flockSystem = new FlockSystem();

await renderer.computeAsync(flockSystem.initializeBoids());

const renderBoids = () => {
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
  material.colorNode = flockSystem.uniforms.boid.colors.element(instanceIndex);
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


  const helper = new THREE.GridHelper(
    flockSystem.gridMax * 2,
    flockSystem.gridWidth / 2,
    0x303030,
    0x303030
  );
  mainScene.scene.add(helper);
};


const clearGrid = flockSystem.clearGrid();
const insertBoidsIntoGrid = flockSystem.insertBoidsIntoGrid();
const computeUpdatePosition = flockSystem.computeUpdatePosition();
const computeFlockPosition = flockSystem.computeFlockPosition();
flockSystem.setupMouse(mainScene.camera);

const updateLoop = async () => {
  await renderer.computeAsync(clearGrid);
  await renderer.computeAsync(insertBoidsIntoGrid);
  await renderer.computeAsync(computeUpdatePosition);
  await renderer.computeAsync(computeFlockPosition);
};

renderBoids();
mainScene.setAnimationLoop(updateLoop);
