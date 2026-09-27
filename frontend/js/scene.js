import * as THREE from "three";

const container = document.getElementById("scene-container");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
scene.fog = new THREE.Fog(0x0d1117, 30, 90);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
camera.position.set(10, 14, 22);
camera.lookAt(10, 0, 6);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const hemi = new THREE.HemisphereLight(0xffffff, 0x223344, 1.0);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 1.1);
sun.position.set(12, 22, 10);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.camera.left = -30;
sun.shadow.camera.right = 30;
sun.shadow.camera.top = 30;
sun.shadow.camera.bottom = -30;
scene.add(sun);

let field = null;
let fieldKey = null;
let stadium = null;
const entityMeshes = new Map();
const entityKinds = new Map(); // id -> "humanoid" | "ball" | "goal" | "plain"
const clock = new THREE.Clock();

function resize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// ---------------------------------------------------------------------------
// Procedural textures (no external assets - everything drawn on a canvas)
// ---------------------------------------------------------------------------

function pitchTexture(fieldW, fieldH) {
  const res = 512;
  const canvas = document.createElement("canvas");
  canvas.width = res;
  canvas.height = Math.round((res * fieldH) / fieldW);
  const ctx = canvas.getContext("2d");

  const stripes = 10;
  const stripeW = canvas.width / stripes;
  for (let i = 0; i < stripes; i++) {
    ctx.fillStyle = i % 2 === 0 ? "#1f7a3d" : "#1a6b35";
    ctx.fillRect(i * stripeW, 0, stripeW, canvas.height);
  }

  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = Math.max(2, res * 0.006);
  const margin = res * 0.03;
  ctx.strokeRect(margin, margin, canvas.width - margin * 2, canvas.height - margin * 2);
  ctx.beginPath();
  ctx.moveTo(canvas.width / 2, margin);
  ctx.lineTo(canvas.width / 2, canvas.height - margin);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(canvas.width / 2, canvas.height / 2, Math.min(canvas.width, canvas.height) * 0.12, 0, Math.PI * 2);
  ctx.stroke();

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function soccerBallTexture() {
  const res = 256;
  const canvas = document.createElement("canvas");
  canvas.width = res;
  canvas.height = res;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#f5f5f5";
  ctx.fillRect(0, 0, res, res);
  ctx.fillStyle = "#1a1a1a";
  const pentagon = (cx, cy, r, rot) => {
    ctx.beginPath();
    for (let i = 0; i < 5; i++) {
      const a = rot + (i * Math.PI * 2) / 5 - Math.PI / 2;
      const x = cx + r * Math.cos(a);
      const y = cy + r * Math.sin(a);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  };
  pentagon(res * 0.5, res * 0.3, res * 0.14, 0);
  pentagon(res * 0.22, res * 0.62, res * 0.13, 0.6);
  pentagon(res * 0.78, res * 0.62, res * 0.13, -0.6);
  pentagon(res * 0.5, res * 0.9, res * 0.12, 0.3);
  pentagon(res * 0.05, res * 0.15, res * 0.1, 1.1);
  pentagon(res * 0.95, res * 0.15, res * 0.1, -1.1);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// ---------------------------------------------------------------------------
// Field + stadium backdrop
// ---------------------------------------------------------------------------

function buildField(fieldSpec) {
  const key = `${fieldSpec.w}x${fieldSpec.h}`;
  if (key === fieldKey) return;
  fieldKey = key;

  if (field) scene.remove(field);
  if (stadium) scene.remove(stadium);

  const group = new THREE.Group();
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(fieldSpec.w, fieldSpec.h),
    new THREE.MeshStandardMaterial({ map: pitchTexture(fieldSpec.w, fieldSpec.h), roughness: 0.9 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(fieldSpec.w / 2, 0, fieldSpec.h / 2);
  ground.receiveShadow = true;
  group.add(ground);
  scene.add(group);
  field = group;

  const cx = fieldSpec.w / 2;
  const cz = fieldSpec.h / 2;
  const radius = Math.max(fieldSpec.w, fieldSpec.h) * 0.85;

  const standsGroup = new THREE.Group();
  const stand = new THREE.Mesh(
    new THREE.TorusGeometry(radius, 1.4, 8, 32),
    new THREE.MeshStandardMaterial({ color: 0x30363d, roughness: 1 })
  );
  stand.rotation.x = Math.PI / 2;
  stand.position.set(cx, 1.4, cz);
  standsGroup.add(stand);

  const lightMat = new THREE.MeshStandardMaterial({ color: 0xfff3b0, emissive: 0xfff3b0, emissiveIntensity: 1.5 });
  const poleMat = new THREE.MeshStandardMaterial({ color: 0x484f58 });
  const corners = [
    [cx - radius * 0.75, cz - radius * 0.75],
    [cx + radius * 0.75, cz - radius * 0.75],
    [cx - radius * 0.75, cz + radius * 0.75],
    [cx + radius * 0.75, cz + radius * 0.75],
  ];
  for (const [x, z] of corners) {
    const poleHeight = 9;
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, poleHeight, 8), poleMat);
    pole.position.set(x, poleHeight / 2, z);
    standsGroup.add(pole);
    const lamp = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.6, 0.3), lightMat);
    lamp.position.set(x, poleHeight, z);
    lamp.lookAt(cx, 0, cz);
    standsGroup.add(lamp);
    const spot = new THREE.PointLight(0xfff3b0, 15, radius * 1.5);
    spot.position.set(x, poleHeight, z);
    standsGroup.add(spot);
  }
  scene.add(standsGroup);
  stadium = standsGroup;

  camera.position.set(cx, Math.max(fieldSpec.w, fieldSpec.h) * 0.75, fieldSpec.h * 1.25);
  camera.lookAt(cx, 0, cz);
}

// ---------------------------------------------------------------------------
// Entity builders
// ---------------------------------------------------------------------------

function buildHumanoid(color) {
  const group = new THREE.Group();
  const skin = new THREE.MeshStandardMaterial({ color: 0xe0ac69, roughness: 0.8 });
  const body = new THREE.MeshStandardMaterial({ color, roughness: 0.7 });

  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.22, 0.35, 4, 8), body);
  torso.position.y = 0.55;
  torso.castShadow = true;
  group.add(torso);

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 12), skin);
  head.position.y = 0.98;
  head.castShadow = true;
  group.add(head);

  const limbGeom = () => new THREE.CapsuleGeometry(0.07, 0.32, 4, 6);

  const makeLimb = (mat, x, y) => {
    const pivot = new THREE.Group();
    pivot.position.set(x, y, 0);
    const mesh = new THREE.Mesh(limbGeom(), mat);
    mesh.position.y = -0.2;
    mesh.castShadow = true;
    pivot.add(mesh);
    group.add(pivot);
    return pivot;
  };

  const legL = makeLimb(body, 0.09, 0.35);
  const legR = makeLimb(body, -0.09, 0.35);
  const armL = makeLimb(skin, 0.26, 0.68);
  const armR = makeLimb(skin, -0.26, 0.68);

  group.userData.limbs = { legL, legR, armL, armR };
  group.userData.walkPhase = 0;
  group.userData.facing = 0;
  return group;
}

function buildBall(radius) {
  const geom = new THREE.SphereGeometry(radius, 20, 20);
  const mat = new THREE.MeshStandardMaterial({ map: soccerBallTexture(), roughness: 0.5 });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  return mesh;
}

function buildGoal(size) {
  const [w, h] = size;
  const group = new THREE.Group();
  const postMat = new THREE.MeshStandardMaterial({ color: 0xf0f0f0, roughness: 0.4 });
  const postRadius = Math.max(0.05, Math.min(w, h) * 0.04);

  const postL = new THREE.Mesh(new THREE.CylinderGeometry(postRadius, postRadius, h, 10), postMat);
  postL.position.set(-w / 2, h / 2, 0);
  postL.castShadow = true;
  group.add(postL);

  const postR = postL.clone();
  postR.position.x = w / 2;
  group.add(postR);

  const bar = new THREE.Mesh(new THREE.CylinderGeometry(postRadius, postRadius, w, 10), postMat);
  bar.rotation.z = Math.PI / 2;
  bar.position.set(0, h, 0);
  bar.castShadow = true;
  group.add(bar);

  const netDepth = Math.min(w, h) * 0.35;
  const netMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.45 });
  const netPoints = [];
  const divisions = 8;
  for (let i = 0; i <= divisions; i++) {
    const x = -w / 2 + (w * i) / divisions;
    netPoints.push(x, h, 0, x, h, -netDepth);
    netPoints.push(x, 0, -netDepth, x, h, -netDepth);
  }
  for (let i = 0; i <= divisions; i++) {
    const y = (h * i) / divisions;
    netPoints.push(-w / 2, y, -netDepth, w / 2, y, -netDepth);
  }
  const netGeom = new THREE.BufferGeometry();
  netGeom.setAttribute("position", new THREE.Float32BufferAttribute(netPoints, 3));
  group.add(new THREE.LineSegments(netGeom, netMat));

  const backPanel = new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.06, side: THREE.DoubleSide })
  );
  backPanel.position.set(0, h / 2, -netDepth);
  group.add(backPanel);

  return group;
}

function buildPlain(entity) {
  let geometry;
  let halfHeight = entity.radius ?? 0.5;
  if (entity.kind === "box") {
    const [sw, sh] = entity.size ?? [1, 1];
    geometry = new THREE.BoxGeometry(sw, sh, sw);
    halfHeight = sh / 2;
  } else {
    geometry = new THREE.SphereGeometry(entity.radius ?? 0.5, 24, 24);
  }
  const material = new THREE.MeshStandardMaterial({ color: entity.color ?? "#ffffff", roughness: 0.7 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.userData.halfHeight = halfHeight;
  return mesh;
}

function buildEntity(entity) {
  const visual = entity.visual ?? "plain";
  if (visual === "humanoid") return buildHumanoid(entity.color ?? "#4f8cff");
  if (visual === "ball") return buildBall(entity.radius ?? 0.35);
  if (visual === "goal") return buildGoal(entity.size ?? [4, 2]);
  return buildPlain(entity);
}

// ---------------------------------------------------------------------------
// Per-tick update
// ---------------------------------------------------------------------------

function animateHumanoid(mesh, velocity, dt) {
  const [vx, , vz] = velocity ?? [0, 0, 0];
  const speed = Math.hypot(vx, vz);
  const { legL, legR, armL, armR } = mesh.userData.limbs;

  if (speed > 0.15) {
    mesh.userData.facing = Math.atan2(vx, vz);
    mesh.userData.walkPhase += Math.min(speed, 8) * dt * 6;
  }
  mesh.rotation.y = mesh.userData.facing;

  const swing = Math.sin(mesh.userData.walkPhase) * Math.min(0.7, 0.15 + speed * 0.05);
  legL.rotation.x = swing;
  legR.rotation.x = -swing;
  armL.rotation.x = -swing;
  armR.rotation.x = swing;
}

function upsertEntity(entity, dt) {
  let mesh = entityMeshes.get(entity.id);
  const visual = entity.visual ?? "plain";
  if (!mesh || entityKinds.get(entity.id) !== visual) {
    if (mesh) scene.remove(mesh);
    mesh = buildEntity(entity);
    scene.add(mesh);
    entityMeshes.set(entity.id, mesh);
    entityKinds.set(entity.id, visual);
  }

  const [x, y, z] = entity.position;
  if (visual === "humanoid") {
    mesh.position.set(x, y, z);
    animateHumanoid(mesh, entity.velocity, dt);
  } else if (visual === "goal") {
    mesh.position.set(x, y, z);
  } else {
    const halfHeight = mesh.userData.halfHeight ?? entity.radius ?? 0.5;
    mesh.position.set(x, halfHeight + y, z);
  }
}

function removeStaleEntities(currentIds) {
  for (const [id, mesh] of entityMeshes) {
    if (!currentIds.has(id)) {
      scene.remove(mesh);
      entityMeshes.delete(id);
      entityKinds.delete(id);
    }
  }
}

export function renderTick(renderState) {
  if (renderState.field) buildField(renderState.field);
  const currentIds = new Set((renderState.entities ?? []).map((e) => e.id));
  removeStaleEntities(currentIds);
  const dt = clock.getDelta();
  for (const entity of renderState.entities ?? []) {
    upsertEntity(entity, dt);
  }
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
