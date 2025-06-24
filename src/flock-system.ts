import {
  float,
  hash,
  instancedArray,
  instanceIndex,
  ShaderNodeObject,
  Fn,
  If,
  mix,
  step,
  atan2,
  cos,
  sin,
  negate,
  vec3,
  rand,
  uv,
  floor,
  add,
  mul,
  vec2,
  atomicStore,
  atomicAdd,
  atomicSub,
  atomicLoad,
  int,
  Loop,
  distance,
  min,
  uniform,
  sub,
} from "three/tsl";
import * as THREE from "three/webgpu";

// Define a type for vec2 based on the vec2 function from three/tsl
type TSLVec2 = ReturnType<typeof vec3>;
type AtomicArray = ReturnType<typeof instancedArray>;

class FlockSystem {
  gridMin: number;
  gridMax: number;
  cellSize: number;
  loopStart: number;
  loopEnd: number;
  gridWidth: number;
  numberOfBuckets: number;
  boidCount: number;
  maxBoidsPerCell: number;
  uniforms!: {
    mouse: {
      position: ShaderNodeObject<
        THREE.UniformNode<THREE.Vector3>
      >;
      isDown: ShaderNodeObject<THREE.UniformNode<number>>;
    };
    grid: {
      cellCounts: AtomicArray;
      cellBoids: AtomicArray;
    };
    boid: {
      size: ShaderNodeObject<THREE.Node>;
      maxSpeed: ShaderNodeObject<THREE.Node>;
      maxForce: ShaderNodeObject<THREE.Node>;
      positions: AtomicArray;
      accelerations: AtomicArray;
      velocities: AtomicArray;
      colors: AtomicArray;
      forces: {
        separation: ShaderNodeObject<THREE.Node>;
        alignment: ShaderNodeObject<THREE.Node>;
        cohesion: ShaderNodeObject<THREE.Node>;
      };
    };
  };

  constructor() {
    this.gridMin = -1000;
    this.gridMax = 1000;
    this.cellSize = 8;
    this.boidCount = 200000;
    this.maxBoidsPerCell = 20; // This is an estimate, adjust as needed
    this.loopStart = -1;
    this.loopEnd = 1;

    this.gridWidth = (this.gridMax - this.gridMin) / this.cellSize;
    this.numberOfBuckets = this.gridWidth * this.gridWidth;

    this.setupUniforms();
  }

  private setupUniforms() {
    const sizeSpeedForce = instancedArray(this.boidCount, "vec3");
    const force = instancedArray(this.boidCount, "vec3");

    this.uniforms = {
      mouse: {
        position: uniform(new THREE.Vector3(0, 0, 0)),
        isDown: uniform(0, "int"),
      },
      grid: {
        cellCounts: instancedArray(this.numberOfBuckets, "int").toAtomic(),
        cellBoids: instancedArray(
          this.numberOfBuckets * this.maxBoidsPerCell,
          "int"
        ).toAtomic(),
      },
      boid: {
        size: sizeSpeedForce.element(instanceIndex).x,
        maxSpeed: sizeSpeedForce.element(instanceIndex).y,
        maxForce: sizeSpeedForce.element(instanceIndex).z,
        positions: instancedArray(this.boidCount, "vec3"),
        accelerations: instancedArray(this.boidCount, "vec3"),
        velocities: instancedArray(this.boidCount, "vec3"),
        colors: instancedArray(this.boidCount, "vec3"),
        forces: {
          separation: force.element(instanceIndex).x,
          alignment: force.element(instanceIndex).y,
          cohesion: force.element(instanceIndex).z,
        },
      },
    };
  }

  initializeBoids() {
    const computeInitialPositions = Fn(() => {
      const position = this.uniforms.boid.positions.element(instanceIndex);
      const velocity = this.uniforms.boid.velocities.element(instanceIndex);

      position.x.assign(
        hash(instanceIndex)
          .mul(this.gridMax * 2)
          .sub(this.gridMax)
      );
      // position.y.assign(hash(instanceIndex.add(2)).mul(60).sub(30)); // -30 to +30
      position.z.assign(
        hash(instanceIndex.add(1))
          .mul(this.gridMax * 2)
          .sub(this.gridMax)
      );

      // Set initial velocities random in all directions
      // random between -10 to +10
      velocity.x.assign(
        hash(instanceIndex.add(2))
          .mul(this.gridMax * 2)
          .sub(this.gridMax)
          .div(this.gridMax)
      );
      velocity.z.assign(
        hash(instanceIndex.add(3))
          .mul(this.gridMax * 2)
          .sub(this.gridMax)
          .div(this.gridMax)
      );

      // set initial size, maxSpeed, maxForce, and behavior forces
      this.uniforms.boid.size.assign(float(0.5));
      this.uniforms.boid.maxSpeed.assign(float(0.7));
      this.uniforms.boid.maxForce.assign(float(0.2));

      this.uniforms.boid.forces.separation.assign(float(0.8));
      this.uniforms.boid.forces.alignment.assign(float(0.3));
      this.uniforms.boid.forces.cohesion.assign(float(0.2));
    })().compute(this.boidCount);

    return computeInitialPositions;
  }

  computeUpdatePosition() {
    const computeUpdatePosition = Fn(() => {
      const position = this.uniforms.boid.positions.element(instanceIndex);
      const velocity = this.uniforms.boid.velocities.element(instanceIndex);
      const acceleration =
        this.uniforms.boid.accelerations.element(instanceIndex);

      const bound = float(this.gridMax);

      // Integrate velocity and position
      velocity.addAssign(acceleration);
      velocity.mulAssign(0.98); // Dampen velocity
      position.addAssign(velocity);
      acceleration.mulAssign(0);

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
    })().compute(this.boidCount);

    return computeUpdatePosition;
  }

  steerTo = Fn<[TSLVec2, TSLVec2]>(([f, vel]) => {
    f.normalizeAssign();
    f.mulAssign(this.uniforms.boid.maxSpeed);
    f.subAssign(vel);
    If(f.length().greaterThan(this.uniforms.boid.maxForce), () => {
      f.normalizeAssign().mulAssign(this.uniforms.boid.maxForce);
    });
    return f;
  });

  computeWander = Fn(() => {
    const currentPos = this.uniforms.boid.positions.element(instanceIndex);
    const currentVel = this.uniforms.boid.velocities.element(instanceIndex);

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
    const desired = target
      .sub(currentPos)
      .normalize()
      .mul(this.uniforms.boid.maxSpeed);
    const steer = desired.sub(currentVel).toVar();

    // Limit to maxForce without branching
    const len = steer.length();
    const limited = steer.normalize().mul(this.uniforms.boid.maxForce);
    const mask = step(this.uniforms.boid.maxForce, len);
    steer.assign(mix(steer, limited, mask));

    return steer;
  });

  computeFlockPosition() {
    const computeFlockPosition = Fn(() => {
      // reset cellCounts each frame
      const currentPos = this.uniforms.boid.positions.element(instanceIndex);
      const currentVel = this.uniforms.boid.velocities.element(instanceIndex);
      const currentAcc =
        this.uniforms.boid.accelerations.element(instanceIndex);

      currentAcc.mulAssign(0);

      const sepSum = vec3().toVar();
      const aliSum = vec3().toVar();
      const cohSum = vec3().toVar();

      const sepCount = float(0).toVar();
      const aliCount = float(0).toVar();
      const cohCount = float(0).toVar();

      const desiredSeparation = this.uniforms.boid.size.mul(4);
      const aliDist = float(8);
      const cohDist = float(6);
      // Get the grid cell for the current boid

      const gridCell = this.getGridCell(currentPos);
      // Check 3x3 grid: from -1 to +1 relative to current cell
      Loop(
        { start: this.loopStart, end: this.loopEnd, condition: "<" },
        ({ i }) => {
          Loop(
            { start: this.loopStart, end: this.loopEnd, condition: "<" },
            ({ i: j }) => {
              // Calculate neighbor cell coordinates
              const nx = gridCell.x.add(i);
              const nz = gridCell.y.add(j);
              // Clamp neighbor coordinates to valid bounds
              const clampedNx = nx.clamp(0, this.gridWidth - 1);
              const clampedNz = nz.clamp(0, this.gridWidth - 1);

              If(
                nx
                  .greaterThanEqual(0)
                  .and(nx.lessThan(this.gridWidth))
                  .and(nz.greaterThanEqual(0))
                  .and(nz.lessThan(this.gridWidth)),
                () => {
                  // cellIdx = nx + nz * gridWidth
                  const cellIdx = add(
                    clampedNx,
                    mul(clampedNz, this.gridWidth)
                  );

                  // Get number of boids in this cell
                  const cellCount = atomicLoad(
                    this.uniforms.grid.cellCounts.element(cellIdx)
                  );
                  const clampedCount = min(
                    cellCount,
                    int(this.maxBoidsPerCell)
                  ).toVar();

                  // Loop through all boids in this cell
                  Loop(
                    { start: 0, end: clampedCount, condition: "<" },
                    ({ i: k }) => {
                      const storageIdx = cellIdx
                        .mul(this.maxBoidsPerCell)
                        .add(k);
                      const otherIdx = atomicLoad(
                        this.uniforms.grid.cellBoids.element(storageIdx)
                      );

                      // Skip if it's the same boid
                      If(otherIdx.notEqual(instanceIndex), () => {
                        const otherPos =
                          this.uniforms.boid.positions.element(otherIdx);
                        const otherVel =
                          this.uniforms.boid.velocities.element(otherIdx);
                        const d = distance(currentPos, otherPos);

                        // Separation
                        If(
                          d
                            .lessThan(desiredSeparation)
                            .and(d.greaterThan(0.01)),
                          () => {
                            const diff = currentPos
                              .sub(otherPos)
                              .normalize()
                              .div(d);
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
            }
          );
        }
      );

      // Apply steer behaviors
      If(sepCount.greaterThan(0), () => {
        sepSum.divAssign(sepCount);
        const steerForce = this.steerTo(sepSum, currentVel);
        sepSum.assign(steerForce.mul(this.uniforms.boid.forces.separation));
        currentAcc.addAssign(sepSum);
      });

      If(aliCount.greaterThan(0), () => {
        aliSum.divAssign(aliCount);

        const steerForce = this.steerTo(aliSum, currentVel);
        aliSum.assign(steerForce.mul(this.uniforms.boid.forces.alignment));
        currentAcc.addAssign(aliSum);
      });

      If(cohCount.greaterThan(0), () => {
        cohSum.divAssign(cohCount);
        cohSum.subAssign(currentPos);
        const steerForce = this.steerTo(cohSum, currentVel);
        cohSum.assign(steerForce.mul(this.uniforms.boid.forces.cohesion));
        currentAcc.addAssign(cohSum);
      });

      // wander force
      const wander = this.computeWander().mul(0.6);
      currentAcc.addAssign(wander);

      // mouse force
      const mouseRange = float(80);
      const mouseDir = this.uniforms.mouse.position.sub(currentPos).normalize();
      const mouseDist = distance(this.uniforms.mouse.position, currentPos);
      If(
        this.uniforms.mouse.isDown.equal(1).and(mouseDist.lessThan(mouseRange)),
        () => {
          const mouseForce = mouseDir.mul(mouseRange.sub(mouseDist)).mul(2);
          mouseForce.divAssign(mouseRange);
          mouseForce.assign(
            this.steerTo(mouseForce, currentVel).mul(
              this.uniforms.boid.maxForce
            )
          );
          currentAcc.subAssign(mouseForce.mul(5));
        }
      );

      // Set color based on density and speed
      const normVel = currentVel.normalize().mul(2);
      const speed = normVel.mul(0.5).add(0.5);
      const neighborCount = sepCount.add(aliCount).add(cohCount);
      const density = neighborCount.div(
        this.maxBoidsPerCell * this.maxBoidsPerCell
      ); 
      const col = vec3(
        density.add(speed.mul(0.8)),
        density,
        density.add(speed)
      );
      this.uniforms.boid.colors.element(instanceIndex).assign(col);
    })().compute(this.boidCount);

    return computeFlockPosition;
  }

  // Grid System
  getGridCellIndex = Fn<[TSLVec2]>(([pos]) => {
    // Normalize coordinates to start from 0
    const normalizedX = pos.x.sub(this.gridMin);
    const normalizedZ = pos.z.sub(this.gridMin);

    const cellX = floor(normalizedX.div(this.cellSize));
    const cellZ = floor(normalizedZ.div(this.cellSize));

    // Clamp to valid grid bounds
    const clampedX = cellX.clamp(0, this.gridWidth - 1);
    const clampedZ = cellZ.clamp(0, this.gridWidth - 1);

    const cell = add(clampedX, mul(clampedZ, this.gridWidth));
    return cell;
  });

  getGridCell = Fn<[TSLVec2]>(([pos]) => {
    // Normalize coordinates to start from 0
    const normalizedX = pos.x.sub(this.gridMin);
    const normalizedZ = pos.z.sub(this.gridMin);

    const cellX = floor(normalizedX.div(this.cellSize));
    const cellZ = floor(normalizedZ.div(this.cellSize));

    // Clamp to valid grid bounds
    const clampedX = cellX.clamp(0, this.gridWidth - 1);
    const clampedZ = cellZ.clamp(0, this.gridWidth - 1);

    return vec2(clampedX, clampedZ);
  });

  clearGrid() {
    const clearGrid = Fn(() => {
      const bucketIndex = instanceIndex;

      // Clear cell count
      If(bucketIndex.lessThan(this.numberOfBuckets), () => {
        atomicStore(this.uniforms.grid.cellCounts.element(bucketIndex), int(0));
      });

      // Clear cell boids
      Loop({ start: 0, end: this.maxBoidsPerCell, condition: "<" }, ({ i }) => {
        const storageIndex = bucketIndex.mul(this.maxBoidsPerCell).add(i);
        If(
          storageIndex.lessThan(this.numberOfBuckets * this.maxBoidsPerCell),
          () => {
            atomicStore(
              this.uniforms.grid.cellBoids.element(storageIndex),
              int(0)
            );
          }
        );
      });
    })().compute(this.numberOfBuckets);

    return clearGrid;
  }

  insertBoidsIntoGrid() {
    const insertBoidsIntoGrid = Fn(() => {
      const boidIndex = instanceIndex;
      const position = this.uniforms.boid.positions.element(boidIndex);
      const cellIndex = this.getGridCellIndex(position);
      const localIndex = atomicAdd(
        this.uniforms.grid.cellCounts.element(cellIndex),
        int(1)
      );

      If(localIndex.lessThan(this.maxBoidsPerCell), () => {
        const storageIndex = cellIndex
          .mul(this.maxBoidsPerCell)
          .add(localIndex);
        atomicStore(
          this.uniforms.grid.cellBoids.element(storageIndex),
          int(boidIndex)
        );
      }).Else(() => {
        atomicSub(this.uniforms.grid.cellCounts.element(cellIndex), int(1));
      });
    })().compute(this.boidCount);

    return insertBoidsIntoGrid;
  }

  setupMouse = (camera: THREE.Camera) => {
    const raycaster = new THREE.Raycaster();
    const mouseNormalized = new THREE.Vector2();
    const mousePlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const intersection = new THREE.Vector3();
    const gridMin = this.gridMin;
    const gridWidth = this.gridWidth;
    const cellSize = this.cellSize;

    // map the interesection to the grid bounds
    function worldToGrid(x: number, _y: number, z: number) {
      const offsetX = x - gridMin;
      const offsetY = 0; // don't need y for 2D grid
      const offsetZ = z - gridMin;

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

    const handleMouseMove = (e: MouseEvent) => {
      // normalized coordinates are in the range of -1 to 1
      mouseNormalized.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouseNormalized.y = -(e.clientY / window.innerHeight) * 2 + 1;
      raycaster.setFromCamera(mouseNormalized, camera);
      raycaster.ray.intersectPlane(mousePlane, intersection);

      const gridCoord = worldToGrid(
        intersection.x,
        intersection.y,
        intersection.z
      );
      const finalX = gridCoord.cellX * cellSize + gridMin + cellSize / 2;
      const finalZ = gridCoord.cellZ * cellSize + gridMin + cellSize / 2;
      this.uniforms.mouse.position.value.x = finalX;
      this.uniforms.mouse.position.value.y = 0;
      this.uniforms.mouse.position.value.z = finalZ;
    };

    document.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("keydown", (e) => {
      if (e.key === "Control") {
        this.uniforms.mouse.isDown.value = 1;
      }
    });
    window.addEventListener("keyup", (e) => {
      if (e.key === "Control") {
        this.uniforms.mouse.isDown.value = 0;
      }
    });
  };

  debugAndPrintInstancedUniform(uniform: AtomicArray) {
    // we need to convert the atomic array to a regular array
    // const array = uniform.toArray();

    // return array.values[0].value;
    return uniform; // Return the uniform for now
  }
}

export { FlockSystem };
