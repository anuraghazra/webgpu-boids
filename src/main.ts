import {
  abs,
  add,
  atan,
  atan2,
  atomicAdd,
  atomicLoad,
  atomicStore,
  atomicSub,
  blendColor,
  Break,
  color,
  Continue,
  cos,
  cross,
  debug,
  distance,
  dot,
  float,
  floor,
  Fn,
  fract,
  hash,
  If,
  instancedArray,
  instancedBufferAttribute,
  instanceIndex,
  int,
  length,
  Loop,
  max,
  min,
  mix,
  mul,
  negate,
  positionLocal,
  rand,
  range,
  select,
  shapeCircle,
  sin,
  smoothstep,
  step,
  storage,
  sub,
  uint,
  uniform,
  uniformGroup,
  uv,
  uv,
  vec2,
  vec3,
  vec4,
  workgroupBarrier,
} from "three/tsl";
import "./style.css";
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/examples/jsm/Addons.js";
import { FlockSystem } from "./flock-system";
import { seededRandom } from "three/src/math/MathUtils.js";
import Stats from "stats-gl";

// Define a type for vec2 based on the vec2 function from three/tsl
type TSLVec2 = ReturnType<typeof vec2>;
const init = async () => {
  const scene = new THREE.Scene();
  const stats = new Stats({
    precision: 3,
    horizontal: false,
    trackGPU: true,
    trackCPT: true,
  });
  document.body.appendChild(stats.dom);

  // Create a camera
  // const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 10000);
  const camera = new THREE.PerspectiveCamera(
    50,
    window.innerWidth / window.innerHeight,
    0.1,
    10000
  );
  camera.position.y = 50;
  // camera.position.y = 100;

  // Create a renderer
  const renderer = new THREE.WebGPURenderer({
    samples: 0,
    antialias: false,
    powerPreference: "high-performance",
  });

  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setAnimationLoop(animate);
  document.body.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 5;
  controls.enableRotate = false;
  // controls.maxDistance = 200;
  controls.update();

  // params
  const gridMin = -1000;
  const gridMax = 1000;
  const cellSize = 6;
  const loopStart = -2;
  const loopEnd = 2;
  const gridWidth = (gridMax - gridMin) / cellSize;
  const numberOfBuckets = gridWidth * gridWidth;
  const boidCount = 200000; // in words - 450k
  const maxBoidsPerCell = 20; // This is an estimate, adjust as needed
  console.log({
    gridMin,
    gridMax,
    cellSize,
    gridWidth,
    numberOfBuckets,
    boidCount,
    maxBoidsPerCell,
  });

  const cellCounts = instancedArray(numberOfBuckets, "int").toAtomic(); // Atomic array to count boids in each cell
  const cellBoids = instancedArray(
    numberOfBuckets * maxBoidsPerCell,
    "int"
  ).toAtomic();

  const sizeSpeedForce = instancedArray(boidCount, "vec3"); // 0.5
  const size = sizeSpeedForce.element(instanceIndex).x; // 0.5
  const maxSpeed = sizeSpeedForce.element(instanceIndex).y; // 0.4
  const maxForce = sizeSpeedForce.element(instanceIndex).z; // 0.2

  const force = instancedArray(boidCount, "vec3");
  const separationForce = force.element(instanceIndex).x; // 0.3
  const alignmentForce = force.element(instanceIndex).y; // 0.2
  const cohesionForce = force.element(instanceIndex).z; // 0.1

  const colors = instancedArray(boidCount, "vec3");
  const positions = instancedArray(boidCount, "vec3");
  const accelerations = instancedArray(boidCount, "vec3");
  const velocities = instancedArray(boidCount, "vec3");
  const mouse = uniform(vec3(0, 0, 0), "vec3");
  const isMouseDown = uniform(0);

  const getGridCell = Fn<[TSLVec2]>(([pos]) => {
    // Normalize coordinates to start from 0
    const normalizedX = pos.x.sub(gridMin);
    const normalizedZ = pos.z.sub(gridMin);

    const cellX = floor(normalizedX.div(cellSize));
    const cellZ = floor(normalizedZ.div(cellSize));

    // Clamp to valid grid bounds
    const clampedX = cellX.clamp(0, gridWidth - 1);
    const clampedZ = cellZ.clamp(0, gridWidth - 1);

    const cell = add(clampedX, mul(clampedZ, gridWidth));
    return cell;
  });

  const getGridCellVec = Fn<[TSLVec2]>(([pos]) => {
    // Normalize coordinates to start from 0
    const normalizedX = pos.x.sub(gridMin);
    const normalizedZ = pos.z.sub(gridMin);

    const cellX = floor(normalizedX.div(cellSize));
    const cellZ = floor(normalizedZ.div(cellSize));

    // Clamp to valid grid bounds
    const clampedX = cellX.clamp(0, gridWidth - 1);
    const clampedZ = cellZ.clamp(0, gridWidth - 1);

    return vec2(clampedX, clampedZ);
  });

  const clearGrid = Fn(() => {
    const bucketIndex = instanceIndex;

    // Clear cell count
    If(bucketIndex.lessThan(numberOfBuckets), () => {
      atomicStore(cellCounts.element(bucketIndex), int(0));
    });

    // Clear cell boids
    Loop({ start: 0, end: maxBoidsPerCell, condition: "<" }, ({ i }) => {
      const storageIndex = bucketIndex.mul(maxBoidsPerCell).add(i);
      If(storageIndex.lessThan(numberOfBuckets * maxBoidsPerCell), () => {
        atomicStore(cellBoids.element(storageIndex), int(0));
      });
    });
  })().compute(numberOfBuckets);

  const insertBoidsIntoGrid = Fn(() => {
    const boidIndex = instanceIndex;
    const position = positions.element(boidIndex);
    const cellIndex = getGridCell(position);
    const localIndex = atomicAdd(cellCounts.element(cellIndex), int(1));

    If(localIndex.lessThan(maxBoidsPerCell), () => {
      const storageIndex = cellIndex.mul(maxBoidsPerCell).add(localIndex);
      atomicStore(cellBoids.element(storageIndex), int(boidIndex));
    }).Else(() => {
      atomicSub(cellCounts.element(cellIndex), int(1));
    });
  })().compute(boidCount);

  // Shader
  const computeInitialPositions = Fn(() => {
    const position = positions.element(instanceIndex);
    const velocity = velocities.element(instanceIndex);
    const color = colors.element(instanceIndex);

    position.x.assign(
      hash(instanceIndex)
        .mul(gridMax * 2)
        .sub(gridMax)
    );
    // position.y.assign(hash(instanceIndex.add(2)).mul(60).sub(30));                  // -30 to +30
    position.z.assign(
      hash(instanceIndex.add(1))
        .mul(gridMax * 2)
        .sub(gridMax)
    );

    // Set initial velocities random in all directions
    // random between -10 to +10
    velocity.x.assign(
      hash(instanceIndex.add(2))
        .mul(gridMax * 2)
        .sub(gridMax)
        .div(gridMax)
    );
    velocity.z.assign(
      hash(instanceIndex.add(3))
        .mul(gridMax * 2)
        .sub(gridMax)
        .div(gridMax)
    );

    // position.x = sub(hash(instanceIndex).mul(gridMax), gridMax / 2);
    // position.y = sub(hash(instanceIndex.add(2)).mul(90), 45);
    // position.z = sub(hash(instanceIndex.add(1)).mul(90), 45);
    // position.z = float(hash(instanceIndex));
    // velocity.z = float(hash(instanceIndex.add(5).mul(2).sub(1)));
    // velocity.x = float(hash(instanceIndex.add(1).mul(2).sub(1)));

    color.r = hash(instanceIndex.add(4));
    color.g = hash(instanceIndex.add(5));
    color.b = hash(instanceIndex.add(6));

    // set initial size, maxSpeed, maxForce, and behavior forces
    size.assign(float(0.5));
    maxSpeed.assign(float(0.9));
    // maxSpeed.assign(float(.2));
    maxForce.assign(float(0.4));

    separationForce.assign(float(0.8));
    alignmentForce.assign(float(0.3));
    cohesionForce.assign(float(0.2));
  })().compute(boidCount);

  const computeUpdatePosition = Fn(() => {
    const position = positions.element(instanceIndex);
    const velocity = velocities.element(instanceIndex);
    const acceleration = accelerations.element(instanceIndex);

    const bound = float(gridMax);

    // Integrate velocity and position
    velocity.addAssign(acceleration);
    // velocity.mulAssign(0.98); // Dampen velocity
    position.addAssign(velocity);
    acceleration.mulAssign(0.99);

    // Bounce X
    // If(position.x.greaterThan(bound), () => {
    //   position.x.assign(bound);
    //   velocity.x.mulAssign(-1);
    // }).Else(() => {
    //   If(position.x.lessThan(bound.negate()), () => {
    //     position.x.assign(bound.negate());
    //     velocity.x.mulAssign(-1);
    //   });
    // });

    // // Bounce Y
    // If(position.y.greaterThan(bound), () => {
    //   position.y.assign(bound);
    //   velocity.y.mulAssign(-1);
    // }).Else(() => {
    //   If(position.y.lessThan(bound.negate()), () => {
    //     position.y.assign(bound.negate());
    //     velocity.y.mulAssign(-1);
    //   });
    // });

    // // Bounce Z
    // If(position.z.greaterThan(bound), () => {
    //   position.z.assign(bound);
    //   velocity.z.mulAssign(-1);
    // }).Else(() => {
    //   If(position.z.lessThan(bound.negate()), () => {
    //     position.z.assign(bound.negate());
    //     velocity.z.mulAssign(-1);
    //   });
    // });

    // Instead of Bounce lets wrap
    If(position.x.greaterThan(bound), () => {
      position.x.assign(position.x.sub(bound.mul(2)));
    }).Else(() => {
      If(position.x.lessThan(bound.negate()), () => {
        position.x.assign(position.x.add(bound.mul(2)));
      });
    });
    If(position.y.greaterThan(bound), () => {
      position.y.assign(position.y.sub(bound.mul(2)));
    }).Else(() => {
      If(position.y.lessThan(bound.negate()), () => {
        position.y.assign(position.y.add(bound.mul(2)));
      });
    });
    If(position.z.greaterThan(bound), () => {
      position.z.assign(position.z.sub(bound.mul(2)));
    }).Else(() => {
      If(position.z.lessThan(bound.negate()), () => {
        position.z.assign(position.z.add(bound.mul(2)));
      });
    });
  })().compute(boidCount);

  const steerTo = Fn<[TSLVec2, TSLVec2]>(([f, vel]) => {
    f.normalizeAssign();
    f.mulAssign(maxSpeed);
    f.subAssign(vel);
    If(f.length().greaterThan(maxForce), () => {
      f.normalizeAssign().mulAssign(maxForce);
    });
    return f;
  });

  const computeWander = Fn(() => {
    const currentPos = positions.element(instanceIndex);
    const currentVel = velocities.element(instanceIndex);

    const wanderR = float(hash(instanceIndex.add(3)).mul(2).add(1));
    const wanderD = float(7);
    const change = float(0.8);

    const baseTheta = rand(uv())
      .mul(2 * Math.PI)
      .add(hash(instanceIndex.mul(0.4)));
    const offsetTheta = rand(vec3(hash(instanceIndex.mul(0.5))))
      .mul(change)
      .sub(change.mul(0.5));
    const theta = baseTheta.add(offsetTheta).toVar();

    const forward = currentVel.normalize();
    const circleLoc = forward.mul(wanderD).add(currentPos);

    // Heading angle for 3D projection
    const h = negate(atan2(negate(currentVel.z), currentVel.x));

    const cosTheta = cos(theta.add(h));
    const sinTheta = sin(theta.add(h));

    const circleOffset = vec3(
      wanderR.mul(cosTheta), // Adjust for 3D projection
      0,
      // wanderR.mul(cosTheta).mul(hash(instanceIndex.mul(4))), // Adjust for 3D projection
      wanderR.mul(sinTheta) // Adjust for 3D projection
    );

    const target = circleLoc.add(circleOffset);
    const desired = target.sub(currentPos).normalize().mul(maxSpeed);
    const steer = desired.sub(currentVel).toVar();

    // Limit to maxForce without branching
    const len = steer.length();
    const limited = steer.normalize().mul(maxForce);
    const mask = step(maxForce, len);
    steer.assign(mix(steer, limited, mask));

    return steer;
  });
  const computeFlockPosition = Fn(() => {
    // reset cellCounts each frame
    const currentPos = positions.element(instanceIndex);
    const currentVel = velocities.element(instanceIndex);
    const currentAcc = accelerations.element(instanceIndex);

    currentAcc.mulAssign(0);

    const sepSum = vec3().toVar();
    const aliSum = vec3().toVar();
    const cohSum = vec3().toVar();

    const sepCount = float(0).toVar();
    const aliCount = float(0).toVar();
    const cohCount = float(0).toVar();

    const desiredSeparation = size.mul(4);
    const aliDist = float(8);
    const cohDist = float(6);
    // Get the grid cell for the current boid

    const gridCell = getGridCellVec(currentPos);
    // Check 3x3 grid: from -1 to +1 relative to current cell
    Loop({ start: loopStart, end: loopEnd, condition: "<" }, ({ i }) => {
      Loop({ start: loopStart, end: loopEnd, condition: "<" }, ({ i: j }) => {
        // Calculate neighbor cell coordinates
        const nx = gridCell.x.add(i);
        const nz = gridCell.y.add(j);
        // Clamp neighbor coordinates to valid bounds
        const clampedNx = nx.clamp(0, gridWidth - 1);
        const clampedNz = nz.clamp(0, gridWidth - 1);

        If(
          nx
            .greaterThanEqual(0)
            .and(nx.lessThan(gridWidth))
            .and(nz.greaterThanEqual(0))
            .and(nz.lessThan(gridWidth)),
          () => {
            // cellIdx = nx + nz * gridWidth
            const cellIdx = add(clampedNx, mul(clampedNz, gridWidth));

            // Get number of boids in this cell
            const cellCount = atomicLoad(cellCounts.element(cellIdx));
            const clampedCount = min(cellCount, int(maxBoidsPerCell)).toVar();

            // Loop through all boids in this cell
            Loop(
              { start: 0, end: clampedCount, condition: "<" },
              ({ i: k }) => {
                const storageIdx = cellIdx.mul(maxBoidsPerCell).add(k);
                const otherIdx = atomicLoad(cellBoids.element(storageIdx));
                
                // Skip if it's the same boid
                If(otherIdx.notEqual(instanceIndex), () => {
                  const otherPos = positions.element(otherIdx);
                  const otherVel = velocities.element(otherIdx);
                  const d = distance(currentPos, otherPos);

                  // Separation
                  If(
                    d.lessThan(desiredSeparation).and(d.greaterThan(0.01)),
                    () => {
                      const diff = currentPos.sub(otherPos).normalize().div(d);
                      sepSum.addAssign(diff);
                      sepCount.addAssign(1);
                    }
                  );

                  // Alignment
                  If(d.lessThan(aliDist), () => {
                    aliSum.addAssign(otherVel);
                    aliCount.addAssign(1);
                  });

                  // Cohesion
                  If(d.lessThan(cohDist), () => {
                    cohSum.addAssign(otherPos);
                    cohCount.addAssign(1);
                  });
                });
              }
            );
          }
        );
      });
    });

    // Apply steer behaviors
    If(sepCount.greaterThan(0), () => {
      sepSum.divAssign(sepCount);
      sepSum.assign(steerTo(sepSum, currentVel).mul(separationForce));
      currentAcc.addAssign(sepSum);
    });

    If(aliCount.greaterThan(0), () => {
      aliSum.divAssign(aliCount);

      aliSum.assign(steerTo(aliSum, currentVel).mul(alignmentForce));
      currentAcc.addAssign(aliSum);
    });

    If(cohCount.greaterThan(0), () => {
      cohSum.divAssign(cohCount);
      cohSum.subAssign(currentPos);
      cohSum.assign(steerTo(cohSum, currentVel).mul(cohesionForce));
      currentAcc.addAssign(cohSum);
    });

    // currentAcc.addAssign(vec3(1, 0, 1).mul(0.1));
    // Wander stays separate
    const wander = computeWander().mul(0.6);
    currentAcc.addAssign(wander);

    // mouse
    const mouseRange = float(80);
    const mouseDir = mouse.sub(currentPos).normalize();
    const mouseDist = distance(mouse, currentPos);
    If(isMouseDown.equal(1).and(mouseDist.lessThan(mouseRange)), () => {
      const mouseForce = mouseDir.mul(mouseRange.sub(mouseDist)).mul(2);
      mouseForce.divAssign(mouseRange);
      mouseForce.assign(steerTo(mouseForce, currentVel).mul(maxForce));
      currentAcc.subAssign(mouseForce.mul(5));
    });

    const normVel = currentVel.normalize().mul(2);
    const speedCol = normVel.mul(0.5).add(0.5);
    const neighborCount = sepCount.add(aliCount).add(cohCount);
    const density = neighborCount.div(maxBoidsPerCell*maxBoidsPerCell); // normalize
    const col = vec3(
      density.add(speedCol.mul(0.8)),
      density,
      density.add(speedCol)
    );
    colors.element(instanceIndex).assign(col);
  })().compute(boidCount);

  // const computeFlockPosition = Fn(() => {
  //   const currentPos = positions.element(instanceIndex);
  //   const currentVel = velocities.element(instanceIndex);
  //   const currentAcc = accelerations.element(instanceIndex);
  //   // const currentMaxSpeed = maxSpeed.element(instanceIndex);

  //   currentAcc.mulAssign(0);

  //   const sepSum = vec3().toVar();
  //   const aliSum = vec3().toVar();
  //   const cohSum = vec3().toVar();

  //   const sepCount = float(0).toVar();
  //   const aliCount = float(0).toVar();
  //   const cohCount = float(0).toVar();

  //   const desiredSeparation = size.mul(2);
  //   const aliDist = float(2);
  //   const cohDist = float(1);

  //   Loop(boidCount, ({ i }) => {
  //     If(instanceIndex.notEqual(i), () => {
  //       const otherPos = positions.element(i);
  //       const otherVel = velocities.element(i);
  //       const d = distance(currentPos, otherPos);
  //       // If the distance is too small, skip this boid

  //       // Separation
  //       If(d.lessThan(desiredSeparation), () => {
  //         const diff = currentPos.sub(otherPos).normalize().div(d);
  //         sepSum.addAssign(diff);
  //         sepCount.addAssign(1);
  //       });

  //       // Alignment
  //       If(d.lessThan(aliDist), () => {
  //         aliSum.addAssign(otherVel);
  //         aliCount.addAssign(1);
  //       });

  //       // Cohesion
  //       If(d.lessThan(cohDist), () => {
  //         cohSum.addAssign(otherPos);
  //         cohCount.addAssign(1);
  //       });
  //     });
  //   });

  //   // Apply steer behaviors
  //   If(sepCount.greaterThan(0), () => {
  //     sepSum.divAssign(sepCount);
  //     sepSum.assign(steerTo(sepSum, currentVel).mul(separationForce));
  //     currentAcc.addAssign(sepSum);
  //   });

  //   If(aliCount.greaterThan(0), () => {
  //     aliSum.divAssign(aliCount);
  //     aliSum.assign(steerTo(aliSum, currentVel).mul(alignmentForce));
  //     currentAcc.addAssign(aliSum);
  //   });

  //   If(cohCount.greaterThan(0), () => {
  //     cohSum.divAssign(cohCount);
  //     cohSum.subAssign(currentPos);
  //     cohSum.assign(steerTo(cohSum, currentVel).mul(cohesionForce));
  //     currentAcc.addAssign(cohSum);
  //   });

  //   // Wander stays separate
  //   const wander = computeWander();
  //   currentAcc.addAssign(wander);
  // })().compute(boidCount);

  const material = new THREE.SpriteNodeMaterial();

  const cameraTarget = controls.target;
  const cameraPosition = uniform(camera.position, "vec3");
  const cameraZoom = uniform(camera.position.distanceTo(cameraTarget), "float");
  const shapeBox = Fn(() => {
    const s = size.mul(1);
    return vec2(s, s);
  });

  const scaleNode = cameraPosition.distance(cameraTarget).mul(0.001).add(size);
  const shapeCircleOrBox = Fn(() => {
    return select(cameraZoom.lessThan(300), shapeCircle(), shapeBox());
  });
  const raycaster = new THREE.Raycaster();
  const mouseNDC = new THREE.Vector2();
  const mousePlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0); // y = 0 plane
  const intersection = new THREE.Vector3();

  function objectSetup() {
    material.colorNode = colors.element(instanceIndex);
    material.positionNode = positions.toAttribute();
    material.scaleNode = scaleNode;
    material.alphaToCoverage = true;
    material.transparent = false;
    material.depthWrite = false;
    // material.blending = THREE.AdditiveBlending;
    material.opacityNode = shapeCircleOrBox();

    const particles = new THREE.Sprite(material);
    particles.count = boidCount;
    particles.frustumCulled = false;
    scene.add(particles);

    const helper = new THREE.GridHelper(
      gridMax * 2,
      gridWidth / 2,
      0x303030,
      0x303030
    );
    // scene.add(helper);

    scene.background = new THREE.Color(0x111111);

    document.addEventListener("mousemove", (event) => {
      mouseNDC.x = (event.clientX / window.innerWidth) * 2 - 1;
      mouseNDC.y = -(event.clientY / window.innerHeight) * 2 + 1;

      raycaster.setFromCamera(mouseNDC, camera);
      raycaster.ray.intersectPlane(mousePlane, intersection);

      // update the uniform
      // map the interesection to the grid bounds
      function worldToGrid(x: number, z: number) {
        const offsetX = x - gridMin;
        const offsetZ = z - gridMin;
        const offsetY = 0; // Not used, but can be set if needed

        const cellX = Math.floor(offsetX / cellSize);
        const cellZ = Math.floor(offsetZ / cellSize);
        const cellY = Math.floor(offsetY / cellSize);

        const clampedX = Math.max(0, Math.min(cellX, gridWidth - 1));
        const clampedZ = Math.max(0, Math.min(cellZ, gridWidth - 1));
        const clampedY = Math.max(0, Math.min(cellY, gridWidth - 1));

        const index =
          clampedX + clampedZ * gridWidth + clampedY * gridWidth * gridWidth;

        return { cellX: clampedX, cellZ: clampedZ, cellY: clampedY, index };
      }

      const x = intersection.x;
      const z = intersection.z;
      const gridCoord = worldToGrid(x, z);
      const finalX = gridCoord.cellX * cellSize + gridMin + cellSize / 2;
      const finalZ = gridCoord.cellZ * cellSize + gridMin + cellSize / 2;
      mouse.value.x = finalX;
      mouse.value.y = gridCoord.cellY;
      mouse.value.z = finalZ;
      // console.log("Mouse Position:", finalX, finalZ);
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Control") {
        isMouseDown.value = 1;
      }
    });
    window.addEventListener("keyup", (e) => {
      if (e.key === "Control") {
        isMouseDown.value = 0;
      }
    });
  }

  objectSetup();
  await renderer.computeAsync(clearGrid);
  await renderer.computeAsync(computeInitialPositions);
  await renderer.computeAsync(insertBoidsIntoGrid);

  // DEBUG
  function debug() {
    // const buffer = new Uint32Array(numberOfBuckets * maxBoidsPerCell);
    // let attribute = new THREE.StorageBufferAttribute(
    //   buffer,
    //   numberOfBuckets * maxBoidsPerCell
    // );
    // const debugBuffer = storage(
    //   attribute,
    //   "uint",
    //   numberOfBuckets * maxBoidsPerCell
    // );

    // const gridCell = Fn(() => {
    //   const cellValue = getGridCell(positions.element(0));
    //   debugBuffer.element(instanceIndex).assign(cellValue);
    // })().compute(boidCount);

    // const t = await renderer.computeAsync(gridCell);
    // const data = await renderer.getArrayBufferAsync(attribute);
    // const arr = new Float32Array(data);
    // console.log("Grid Cell:", arr[0], arr[1], arr[2]);
    const debugPositions = Fn(() => {
      const pos = positions.element(instanceIndex);
      // Set a simple test position if not initialized
      If(pos.length().lessThan(0.1), () => {
        pos.x.assign(hash(instanceIndex).mul(20).sub(10)); // Random -10 to +10
        pos.y.assign(0);
        pos.z.assign(hash(instanceIndex.add(1000)).mul(20).sub(10));
      });
    })().compute(boidCount);

    // DEBUG: Test grid cell calculation
    const debugGridCells = Fn(() => {
      const pos = positions.element(instanceIndex);
      const cellIdx = getGridCell(pos);

      // Force positions to be within grid if they're outside
      If(
        cellIdx.lessThan(0).or(cellIdx.greaterThanEqual(numberOfBuckets)),
        () => {
          pos.x.assign(pos.x.clamp(-119, 119)); // Keep within grid bounds
          pos.z.assign(pos.z.clamp(-119, 119));
        }
      );
    })().compute(boidCount);

    // DEBUG: Simple grid insertion test
    const debugInsertBoids = Fn(() => {
      const boidIndex = instanceIndex;
      const position = positions.element(boidIndex);
      const cellIndex = getGridCell(position);

      // Only insert if cell index is valid
      If(
        cellIndex.greaterThanEqual(0).and(cellIndex.lessThan(numberOfBuckets)),
        () => {
          const localIndex = atomicAdd(cellCounts.element(cellIndex), int(1));

          If(localIndex.lessThan(maxBoidsPerCell), () => {
            const storageIndex = cellIndex.mul(maxBoidsPerCell).add(localIndex);
            atomicStore(cellBoids.element(storageIndex), int(boidIndex));
          }).Else(() => {
            // Handle overflow - decrement count since we couldn't store
            atomicSub(cellCounts.element(cellIndex), int(1));
          });
        }
      );
    })().compute(boidCount);

    // Add this to your objectSetup() function:
    async function debugGridSystem() {
      console.log("Starting grid debug...");

      await renderer.computeAsync(debugPositions);
      await renderer.computeAsync(debugGridCells);
      await renderer.computeAsync(clearGrid);
      await renderer.computeAsync(debugInsertBoids);

      // Check if any cells have boids
      const cellCountBuffer = new Uint32Array(numberOfBuckets);
      const cellCountAttribute = new THREE.StorageBufferAttribute(
        cellCountBuffer,
        1
      );
      const cellCountStorage = storage(
        cellCountAttribute,
        "uint",
        numberOfBuckets
      );

      const readCellCounts = Fn(() => {
        cellCountStorage
          .element(instanceIndex)
          .assign(atomicLoad(cellCounts.element(instanceIndex)));
      })().compute(numberOfBuckets);

      await renderer.computeAsync(readCellCounts);
      const data = await renderer.getArrayBufferAsync(cellCountAttribute);
      const counts = new Uint32Array(data);

      const nonZeroCells = counts.filter((c) => c > 0).length;
      const totalBoids = counts.reduce((sum, c) => sum + c, 0);

      console.log(`Non-zero cells: ${nonZeroCells}/${numberOfBuckets}`);
      console.log(`Total boids in grid: ${totalBoids}/${boidCount}`);
      console.log(`Max boids in a cell: ${Math.max(...counts)}`);

      if (totalBoids === 0) {
        console.error("NO BOIDS INSERTED INTO GRID!");
      }
    }
  }
  // End DEBUG
  async function animate() {
    stats.update();
    controls.update();

    cameraZoom.value = camera.position.distanceTo(controls.target);
    await renderer.computeAsync(clearGrid);
    await renderer.computeAsync(insertBoidsIntoGrid);
    await renderer.computeAsync(computeUpdatePosition);
    await renderer.computeAsync(computeFlockPosition);

    // Render the scene from the perspective of the camera
    await renderer.renderAsync(scene, camera);
  }
};

init();
