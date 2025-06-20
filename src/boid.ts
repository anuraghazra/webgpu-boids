// const normalBoid = new Boid({
//   name: "normal",
//   id: 0,
//   size: 0.5,
//   color: color(0.5, 0.5, 0.5),
//   maxSpeed: 0.4,
//   maxForce: 0.2,
//   behavior: {
//     separation: 0.3,
//     alignment: 0.2,
//     cohesion: 0.1,
//   },
// });

// const predatorBoid = new Boid({
//   name: "predator",
//   id: 1,
//   size: 0.5,
//   color: color(1, 0, 0),
//   maxSpeed: 0.6,
//   maxForce: 0.3,
//   behavior: {
//     separation: 0.6,
//     alignment: 0.4,
//     cohesion: 0.2,
//   },
// });

// const flock = new FlockSystem({
//   count: 15000,
// });
// flock.addSpecies({
//   id: 0,
//   boid: normalBoid,
//   population: 0.8,
// });
// flock.addSpecies({
//   id: 1,
//   boid: predatorBoid,
//   population: 0.2,
// });
// flock.addRelationship({
//   from: 0,
//   to: 1,
//   type: "flee",
//   distance: 10,
//   strength: 0.5,
// });
// flock.addRelationship({
//   from: 1,
//   to: 0,
//   type: "seek",
//   distance: 10,
//   strength: 0.5,
// });

class Boid {}
