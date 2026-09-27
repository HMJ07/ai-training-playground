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
let currentFieldSpec = null;
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

function roomFloorTexture(fieldW, fieldH) {
  const res = 512;
  const canvas = document.createElement("canvas");
  canvas.width = res;
  canvas.height = Math.round((res * fieldH) / fieldW);
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#8a6d4f";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const tileSize = Math.max(8, res / (fieldW * 1.2));
  for (let y = 0; y < canvas.height; y += tileSize) {
    for (let x = 0; x < canvas.width; x += tileSize) {
      const shade = (Math.floor(x / tileSize) + Math.floor(y / tileSize)) % 2 === 0 ? 10 : -6;
      ctx.fillStyle = `rgba(0,0,0,${shade > 0 ? shade / 255 : 0})`;
      if (shade < 0) ctx.fillStyle = `rgba(255,255,255,${-shade / 255})`;
      ctx.fillRect(x, y, tileSize, tileSize);
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function plainFloorTexture(fieldW, fieldH) {
  const res = 512;
  const canvas = document.createElement("canvas");
  canvas.width = res;
  canvas.height = Math.round((res * fieldH) / fieldW);
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#1c2530";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  const step = Math.max(16, res / (fieldW * 2));
  for (let x = 0; x <= canvas.width; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y <= canvas.height; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function dataFloorTexture(fieldW, fieldH) {
  const res = 512;
  const canvas = document.createElement("canvas");
  canvas.width = res;
  canvas.height = Math.round((res * fieldH) / fieldW);
  const ctx = canvas.getContext("2d");

  const grad = ctx.createRadialGradient(
    canvas.width / 2,
    canvas.height / 2,
    0,
    canvas.width / 2,
    canvas.height / 2,
    Math.max(canvas.width, canvas.height) / 1.4
  );
  grad.addColorStop(0, "#111a2b");
  grad.addColorStop(1, "#05070c");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(90,160,255,0.18)";
  ctx.lineWidth = 1;
  const step = Math.max(14, res / (fieldW * 1.6));
  for (let x = 0; x <= canvas.width; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y <= canvas.height; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function glowSpriteTexture() {
  const res = 128;
  const canvas = document.createElement("canvas");
  canvas.width = res;
  canvas.height = res;
  const ctx = canvas.getContext("2d");
  const grad = ctx.createRadialGradient(res / 2, res / 2, 0, res / 2, res / 2, res / 2);
  grad.addColorStop(0, "rgba(255,255,255,0.9)");
  grad.addColorStop(0.4, "rgba(255,255,255,0.35)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, res, res);
  return new THREE.CanvasTexture(canvas);
}
const GLOW_SPRITE_TEXTURE = glowSpriteTexture();

// ---------------------------------------------------------------------------
// Field + stadium backdrop
// ---------------------------------------------------------------------------

const GROUND_TEXTURE_BY_THEME = {
  pitch: pitchTexture,
  room: roomFloorTexture,
  plain: plainFloorTexture,
  data: dataFloorTexture,
};

function buildField(fieldSpec) {
  currentFieldSpec = fieldSpec;
  const theme = fieldSpec.theme ?? "plain";
  const key = `${fieldSpec.w}x${fieldSpec.h}:${theme}`;
  if (key === fieldKey) return;
  fieldKey = key;

  if (field) scene.remove(field);
  if (stadium) scene.remove(stadium);

  const textureFn = GROUND_TEXTURE_BY_THEME[theme] ?? plainFloorTexture;
  const roughnessByTheme = { pitch: 0.9, room: 0.6, plain: 0.6, data: 0.25 };
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(fieldSpec.w, fieldSpec.h),
    new THREE.MeshStandardMaterial({
      map: textureFn(fieldSpec.w, fieldSpec.h),
      roughness: roughnessByTheme[theme] ?? 0.6,
      metalness: theme === "data" ? 0.3 : 0,
    })
  );
  const group = new THREE.Group();
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(fieldSpec.w / 2, 0, fieldSpec.h / 2);
  ground.receiveShadow = true;
  group.add(ground);

  const cx = fieldSpec.w / 2;
  const cz = fieldSpec.h / 2;

  if (theme === "room") {
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x2a3038, roughness: 0.95 });
    const wallHeight = 4;
    const back = new THREE.Mesh(new THREE.PlaneGeometry(fieldSpec.w, wallHeight), wallMat);
    back.position.set(cx, wallHeight / 2, 0);
    group.add(back);
    const left = new THREE.Mesh(new THREE.PlaneGeometry(fieldSpec.h, wallHeight), wallMat);
    left.rotation.y = Math.PI / 2;
    left.position.set(0, wallHeight / 2, cz);
    group.add(left);
  }

  scene.add(group);
  field = group;

  if (theme === "pitch") {
    stadium = buildStadium(cx, cz, fieldSpec);
    scene.add(stadium);
  } else {
    stadium = null;
  }

  // Pull back based on whichever dimension is larger, not just height, so
  // wide-but-shallow fields (e.g. a 30x20 penalty box) still fit in frame.
  const span = Math.max(fieldSpec.w, fieldSpec.h);
  camera.position.set(cx, span * 0.95, cz + span * 0.95);
  camera.lookAt(cx, 0, cz);
}

function buildStadium(cx, cz, fieldSpec) {
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
  return standsGroup;
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

// `size` is the entity's 2D ground footprint [x-extent, y-extent] - the same
// numbers the physics engine uses (pymunk has no concept of "3D height").
// A goal is thin along one axis (the goal line) and long along the other
// (the mouth) - so the opening width is whichever number is bigger, and we
// orient + face the goal from that, rather than assuming a fixed layout.
function buildGoal(size) {
  const w = Math.max(size[0], size[1], 1);
  const h = 2.2; // real-world-ish goal height; footprint carries no vertical info
  const group = new THREE.Group();
  const postMat = new THREE.MeshStandardMaterial({ color: 0xf0f0f0, roughness: 0.4 });
  const postRadius = Math.max(0.05, w * 0.02);

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

  const netDepth = Math.max(0.6, Math.min(w * 0.3, h * 0.8));
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

// The goal above is built "facing +Z" by default (mouth open toward +Z,
// net receding toward -Z), with posts spread along local X using the
// footprint's longer side. This works out which way to actually turn it so
// the mouth faces the middle of the field, from nothing but its own
// footprint shape and where it sits relative to the field center.
function goalFacingRotationY(entity, fieldSpec) {
  const [sx, sy] = entity.size ?? [4, 1];
  let rotY = sy > sx ? Math.PI / 2 : 0;

  if (!fieldSpec) return rotY;
  const [ex, , ez] = entity.position;
  const outwardX = ex - fieldSpec.w / 2;
  const outwardZ = ez - fieldSpec.h / 2;

  // Local -Z (the net's side) expressed in world space after rotating by rotY.
  const netFacingX = -Math.sin(rotY);
  const netFacingZ = -Math.cos(rotY);
  const netPointsOutward = netFacingX * outwardX + netFacingZ * outwardZ > 0;
  if (!netPointsOutward) rotY += Math.PI;
  return rotY;
}

function buildMarker(entity) {
  const [sx, sy] = entity.size ?? [entity.radius ? entity.radius * 2 : 1, entity.radius ? entity.radius * 2 : 1];
  const radius = Math.max(sx, sy) / 2 || entity.radius || 0.6;

  const group = new THREE.Group();
  const disc = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, 0.03, 24),
    new THREE.MeshStandardMaterial({
      color: entity.color ?? "#ffd166",
      emissive: entity.color ?? "#ffd166",
      emissiveIntensity: 0.6,
      transparent: true,
      opacity: 0.55,
    })
  );
  disc.position.y = 0.02;
  group.add(disc);

  const ringGeom = new THREE.RingGeometry(radius * 0.92, radius, 32);
  const ring = new THREE.Mesh(
    ringGeom,
    new THREE.MeshBasicMaterial({ color: entity.color ?? "#ffd166", side: THREE.DoubleSide })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.04;
  group.add(ring);

  group.userData.halfHeight = 0;
  return group;
}

// A classification example: a small glowing orb (a data point in embedding
// space, not a physical object) plus a soft billboard halo and a thin beam
// down to the floor so it reads as "floating above the grid" rather than a
// bare sphere sitting in a void.
function buildDatapoint(entity) {
  const radius = entity.radius ?? 0.35;
  const color = entity.color ?? "#4f8cff";
  const group = new THREE.Group();

  const core = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 16, 16),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.8, roughness: 0.3 })
  );
  group.add(core);

  const glow = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: GLOW_SPRITE_TEXTURE,
      color,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
  );
  glow.scale.set(radius * 6, radius * 6, 1);
  group.add(glow);

  const beamMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.25 });
  const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, 1, 6), beamMat);
  group.add(beam);
  group.userData.beam = beam;

  group.userData.halfHeight = 0;
  return group;
}

function buildPlain(entity) {
  let geometry;
  let halfHeight = entity.radius ?? 0.5;
  if (entity.kind === "box") {
    // `size` is the 2D ground footprint [x-extent, y-extent], same as the
    // physics box - it carries no vertical info, so we derive a sensible
    // visual height from the footprint rather than misreading size[1] as one.
    const [sx, sy] = entity.size ?? [1, 1];
    const visualHeight = Math.min(3, Math.max(0.5, Math.min(sx, sy) * 1.2));
    geometry = new THREE.BoxGeometry(sx, visualHeight, sy);
    halfHeight = visualHeight / 2;
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
  if (visual === "marker") return buildMarker(entity);
  if (visual === "datapoint") return buildDatapoint(entity);
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
    mesh.rotation.y = goalFacingRotationY(entity, currentFieldSpec);
  } else if (visual === "datapoint") {
    mesh.position.set(x, y, z);
    const beam = mesh.userData.beam;
    if (beam && y > 0.05) {
      beam.scale.y = y;
      beam.position.y = -y / 2;
    } else if (beam) {
      beam.visible = false;
    }
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
