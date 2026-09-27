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
let fieldKey = null;
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
  const key = `${fieldSpec.w}x${fieldSpec.h}`;
  if (key === fieldKey) return;
  fieldKey = key;

  if (field) scene.remove(field);
  const group = new THREE.Group();

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(fieldSpec.w, fieldSpec.h),
    new THREE.MeshStandardMaterial({ color: 0x14532d })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(fieldSpec.w / 2, 0, fieldSpec.h / 2);
  group.add(ground);

  scene.add(group);
  field = group;
  camera.position.set(fieldSpec.w / 2, Math.max(fieldSpec.w, fieldSpec.h) * 0.9, fieldSpec.h * 1.3);
  camera.lookAt(fieldSpec.w / 2, 0, fieldSpec.h / 2);
}

function upsertEntity(entity) {
  let mesh = entityMeshes.get(entity.id);
  if (!mesh) {
    let geometry;
    let halfHeight = entity.radius ?? 0.5;
    if (entity.kind === "box") {
      const [sw, sh] = entity.size ?? [1, 1];
      geometry = new THREE.BoxGeometry(sw, sh, sw);
      halfHeight = sh / 2;
    } else {
      geometry = new THREE.SphereGeometry(entity.radius ?? 0.5, 24, 24);
    }
    const material = new THREE.MeshStandardMaterial({ color: entity.color ?? "#ffffff" });
    mesh = new THREE.Mesh(geometry, material);
    mesh.userData.halfHeight = halfHeight;
    scene.add(mesh);
    entityMeshes.set(entity.id, mesh);
  }
  const [x, y, z] = entity.position;
  mesh.position.set(x, mesh.userData.halfHeight + y, z);
}

function removeStaleEntities(currentIds) {
  for (const [id, mesh] of entityMeshes) {
    if (!currentIds.has(id)) {
      scene.remove(mesh);
      entityMeshes.delete(id);
    }
  }
}

export function renderTick(renderState) {
  if (renderState.field) buildField(renderState.field);
  const currentIds = new Set((renderState.entities ?? []).map((e) => e.id));
  removeStaleEntities(currentIds);
  for (const entity of renderState.entities ?? []) {
    upsertEntity(entity);
  }
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
