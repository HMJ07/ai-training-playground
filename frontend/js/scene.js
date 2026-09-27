import * as THREE from "three";

const container = document.getElementById("scene-container");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
camera.position.set(10, 14, 22);
camera.lookAt(10, 0, 6);

const renderer = new THREE.WebGLRenderer({ antialias: true });
container.appendChild(renderer.domElement);

const light = new THREE.HemisphereLight(0xffffff, 0x223344, 1.1);
scene.add(light);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(5, 10, 5);
scene.add(dirLight);

let field = null;
const entityMeshes = new Map();

function resize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function buildField(fieldSpec) {
  if (field) scene.remove(field);
  const group = new THREE.Group();

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(fieldSpec.w, fieldSpec.h),
    new THREE.MeshStandardMaterial({ color: 0x14532d })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(fieldSpec.w / 2, 0, fieldSpec.h / 2);
  group.add(ground);

  const [gy0, gy1] = fieldSpec.goalYRange;
  const goal = new THREE.Mesh(
    new THREE.BoxGeometry(0.2, 1.5, gy1 - gy0),
    new THREE.MeshStandardMaterial({ color: 0xffd166 })
  );
  goal.position.set(fieldSpec.w, 0.75, (gy0 + gy1) / 2);
  group.add(goal);

  scene.add(group);
  field = group;
  camera.lookAt(fieldSpec.w / 2, 0, fieldSpec.h / 2);
}

function upsertEntity(entity) {
  let mesh = entityMeshes.get(entity.id);
  if (!mesh) {
    let geometry;
    if (entity.kind === "sphere") {
      geometry = new THREE.SphereGeometry(entity.radius ?? 0.5, 24, 24);
    } else {
      geometry = new THREE.BoxGeometry(1, 1, 1);
    }
    const material = new THREE.MeshStandardMaterial({ color: entity.color ?? "#ffffff" });
    mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    entityMeshes.set(entity.id, mesh);
  }
  const [x, y, z] = entity.position;
  mesh.position.set(x, (entity.radius ?? 0.5) + y, z);
}

export function renderTick(renderState) {
  if (renderState.field) buildField(renderState.field);
  for (const entity of renderState.entities ?? []) {
    upsertEntity(entity);
  }
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
