// src/utils/arkitsToExpression.ts
// ARkit blendshape → VRM 表情转换器
// 当 ARkit 值超过阈值时，计算衍生表情并映射到 VRM ShapeKey
//
// 输出两种格式的 key 以兼容不同 VRM 模型：
//   - facial1.brow_angry_L  (VRoid Studio 带材质前缀)
//   - brow_angry_L          (不带前缀)

/** 单次转换结果：衍生表情名 → 值 */
export type ExpressionResult = Record<string, number>;

/** 衍生表情的基础 ShapeKey 名（不含 facial1. 前缀） */
const EXPRESSION_KEYS = {
  angry: ['brow_angry_L', 'brow_angry_R', 'eye_jito_L', 'eye_jito_R', 'mouth_angry'] as const,
  joy: ['eye_smile_L', 'eye_smile_R'] as const,
  sorrow: ['brow_sad_L', 'brow_sad_R', 'eye_jito_L', 'eye_jito_R', 'mouth_angry'] as const,
  surprise: ['eye_surprise', 'mouth_angry'] as const,
  puku: ['cheek_puff'] as const,
  jitome: ['eye_jito_L', 'eye_jito_R'] as const,
  tired: ['eye_jito_L', 'eye_jito_R', 'brow_sad_L', 'brow_sad_R', 'mouth_angry'] as const,
} as const;

/**
 * 将基础 key 注册到结果中（同时写入带/不带 facial1. 前缀的版本）
 */
function setExpr(result: ExpressionResult, baseKey: string, value: number): void {
  result[baseKey] = value;
  result['facial1.' + baseKey] = value;
}

/**
 * 从 ARkit blendshapes 计算所有衍生表情
 * @returns { [ShapeKey名]: 权重值 } — 包含两种格式 key 以兼容不同模型
 */
export function computeExpressions(arkit: Record<string, number>): ExpressionResult {
  const result: ExpressionResult = {};

  // ── Angry（生气） ──
  const browDownLeft = arkit.browDownLeft ?? 0;
  const browDownRight = arkit.browDownRight ?? 0;
  if (browDownLeft > 0.40 && browDownRight > 0.40) {
    const brow = Math.min(browDownLeft, browDownRight);
    const blink = ((arkit.eyeBlinkLeft ?? 0) + (arkit.eyeBlinkRight ?? 0)) / 2;
    const press = ((arkit.mouthPressLeft ?? 0) + (arkit.mouthPressRight ?? 0)) / 2;
    const angry = 0.55 * brow + 0.25 * blink + 0.20 * press;
    if (angry > 0.45) {
      for (const k of EXPRESSION_KEYS.angry) setExpr(result, k, angry);
    }
  }

  // ── Joy（笑） ──
  const smileMax = Math.max(arkit.mouthSmileLeft ?? 0, arkit.mouthSmileRight ?? 0);
  if (smileMax > 0.30) {
    const smile = ((arkit.mouthSmileLeft ?? 0) + (arkit.mouthSmileRight ?? 0)) / 2;
    const blink = ((arkit.eyeBlinkLeft ?? 0) + (arkit.eyeBlinkRight ?? 0)) / 2;
    const joy = 0.70 * smile + 0.30 * blink;
    if (joy > 0.40) {
      for (const k of EXPRESSION_KEYS.joy) setExpr(result, k, joy);
    }
  }

  // ── Sorrow（伤心） ──
  const browInnerUp = arkit.browInnerUp ?? 0;
  if (browInnerUp > 0.30) {
    const blink = ((arkit.eyeBlinkLeft ?? 0) + (arkit.eyeBlinkRight ?? 0)) / 2;
    const press = ((arkit.mouthPressLeft ?? 0) + (arkit.mouthPressRight ?? 0)) / 2;
    const sorrow = 0.50 * browInnerUp + 0.30 * blink + 0.20 * press;
    if (sorrow > 0.40) {
      for (const k of EXPRESSION_KEYS.sorrow) setExpr(result, k, sorrow);
    }
  }

  // ── Surprise（惊讶） ──
  const eyeWideLeft = arkit.eyeWideLeft ?? 0;
  const eyeWideRight = arkit.eyeWideRight ?? 0;
  const jawOpen = arkit.jawOpen ?? 0;
  if (eyeWideLeft > 0.35 || eyeWideRight > 0.35 || jawOpen > 0.35) {
    const eye = (eyeWideLeft + eyeWideRight) / 2;
    const surprise = 0.80 * eye + 0.20 * jawOpen;
    if (surprise > 0.40) {
      for (const k of EXPRESSION_KEYS.surprise) setExpr(result, k, surprise);
    }
  }

  // ── Puku（鼓脸） ──
  const cheekPuff = arkit.cheekPuff ?? 0;
  if (cheekPuff > 0.30) {
    for (const k of EXPRESSION_KEYS.puku) setExpr(result, k, cheekPuff);
  }

  // ── Jitome（嫌弃眼） ──
  const blinkAvg = ((arkit.eyeBlinkLeft ?? 0) + (arkit.eyeBlinkRight ?? 0)) / 2;
  if (blinkAvg > 0.20 && blinkAvg < 0.80) {
    const jitome = blinkAvg * 0.60;
    if (jitome > 0.20) {
      for (const k of EXPRESSION_KEYS.jitome) setExpr(result, k, jitome);
    }
  }

  // ── Tired（疲惫） ──
  if (browInnerUp > 0.25 && blinkAvg > 0.25) {
    const press = ((arkit.mouthPressLeft ?? 0) + (arkit.mouthPressRight ?? 0)) / 2;
    const tired = 0.40 * blinkAvg + 0.40 * browInnerUp + 0.20 * press;
    if (tired > 0.35) {
      for (const k of EXPRESSION_KEYS.tired) setExpr(result, k, tired);
    }
  }

  return result;
}
