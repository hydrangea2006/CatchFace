// src/components/LocalArkitsSimulator.tsx
// 本地 ARkit 模拟数据源——无需后端即可测试表情驱动效果
// 交叉淡入淡出：表情之间平滑过渡，无"闪回"/"闪现"间隙
import React, { useEffect, useRef } from 'react';
import type { ArkitsFrame } from './VrmModel';

interface Props {
  frameRef: React.RefObject<ArkitsFrame | null>;
  enabled: boolean;
}

/** 完整的 ARkit blendshapes 默认模板（52 个参数，全部为 0） */
function blankBlendshapes(): Record<string, number> {
  return {
    eyeBlinkLeft: 0, eyeBlinkRight: 0,
    eyeLookDownLeft: 0, eyeLookDownRight: 0,
    eyeLookInLeft: 0, eyeLookInRight: 0,
    eyeLookOutLeft: 0, eyeLookOutRight: 0,
    eyeLookUpLeft: 0, eyeLookUpRight: 0,
    eyeSquintLeft: 0, eyeSquintRight: 0,
    eyeWideLeft: 0, eyeWideRight: 0,
    browDownLeft: 0, browDownRight: 0,
    browInnerUp: 0,
    browOuterUpLeft: 0, browOuterUpRight: 0,
    jawOpen: 0,
    jawForward: 0, jawLeft: 0, jawRight: 0,
    mouthClose: 0, mouthFunnel: 0, mouthPucker: 0,
    mouthLeft: 0, mouthRight: 0,
    mouthSmileLeft: 0, mouthSmileRight: 0,
    mouthFrownLeft: 0, mouthFrownRight: 0,
    mouthStretchLeft: 0, mouthStretchRight: 0,
    mouthDimpleLeft: 0, mouthDimpleRight: 0,
    mouthPressLeft: 0, mouthPressRight: 0,
    mouthRollLower: 0, mouthRollUpper: 0,
    mouthShrugLower: 0, mouthShrugUpper: 0,
    mouthUpperUpLeft: 0, mouthUpperUpRight: 0,
    mouthLowerDownLeft: 0, mouthLowerDownRight: 0,
    cheekPuff: 0, cheekSquintLeft: 0, cheekSquintRight: 0,
    noseSneerLeft: 0, noseSneerRight: 0,
    tongueOut: 0,
  };
}

/** 表情预设 */
const EXPRESSION_PRESETS: Array<{ name: string; bs: Record<string, number> }> = [
  {
    name: '😐 默认',
    bs: {},
  },
  {
    name: '😉 双眼紧闭',
    bs: { eyeBlinkLeft: 1.0, eyeBlinkRight: 1.0 },
  },
  {
    name: '😮 大张嘴',
    bs: { jawOpen: 1.0, mouthFunnel: 0.3 },
  },
  {
    name: '😊 大笑',
    bs: {
      mouthSmileLeft: 1.0, mouthSmileRight: 1.0,
      mouthDimpleLeft: 0.5, mouthDimpleRight: 0.5,
      cheekSquintLeft: 0.4, cheekSquintRight: 0.4,
      eyeBlinkLeft: 0.2, eyeBlinkRight: 0.2,
    },
  },
  {
    name: '😡 生气',
    bs: {
      browDownLeft: 1.0, browDownRight: 1.0,
      eyeBlinkLeft: 0.3, eyeBlinkRight: 0.3,
      mouthPressLeft: 0.6, mouthPressRight: 0.6,
    },
  },
  {
    name: '😢 伤心',
    bs: {
      browInnerUp: 0.9,
      eyeBlinkLeft: 0.4, eyeBlinkRight: 0.4,
      mouthPressLeft: 0.5, mouthPressRight: 0.5,
    },
  },
  {
    name: '😲 惊讶',
    bs: {
      eyeWideLeft: 1.0, eyeWideRight: 1.0,
      jawOpen: 0.7,
    },
  },
  {
    name: '😗 噘嘴嘟嘟',
    bs: { mouthPucker: 1.0, jawOpen: 0.15 },
  },
  {
    name: '😴 疲惫',
    bs: {
      browInnerUp: 0.6,
      eyeBlinkLeft: 0.6, eyeBlinkRight: 0.6,
      mouthPressLeft: 0.4, mouthPressRight: 0.4,
    },
  },
  {
    name: '😒 嫌弃眼',
    bs: {
      eyeBlinkLeft: 0.5, eyeBlinkRight: 0.5,
    },
  },
  {
    name: '🐡 鼓脸',
    bs: { cheekPuff: 1.0 },
  },
];

const HOLD_DURATION = 2000;      // 表情稳定保持 ms
const TRANSITION_DURATION = 600;  // 淡入淡出过渡 ms

const LocalArkitsSimulator: React.FC<Props> = ({ frameRef, enabled }) => {
  const frameIdRef = useRef<number>(0);

  useEffect(() => {
    if (!enabled) return;

    let currentIndex = 0;
    let switchTime = performance.now();
    // 上一帧实际输出——用于平滑 lerp
    const prevBs = blankBlendshapes();
    // 当前"稳定态"的目标值——切换后作为 lerp 的起点
    const stableBs = blankBlendshapes();
    let debugOnce = true;

    console.log('[LocalArkitsSimulator] 🚀 启动本地模拟模式');
    console.log(
      '[LocalArkitsSimulator] 预设序列:',
      EXPRESSION_PRESETS.map((p, i) => `${i}:${p.name}`).join(' → ')
    );

    const loop = (now: number) => {
      const elapsed = now - switchTime;

      // 稳定阶段结束 → 切换到下一个预设
      if (elapsed >= HOLD_DURATION + TRANSITION_DURATION) {
        // 保存当前稳定态的值
        const currPreset = EXPRESSION_PRESETS[currentIndex];
        for (const key of Object.keys(stableBs)) {
          stableBs[key] = currPreset.bs[key] || 0;
        }

        currentIndex = (currentIndex + 1) % EXPRESSION_PRESETS.length;
        switchTime = now;
        debugOnce = true;
      }

      const reElapsed = now - switchTime; // 重新计算（切换后归零）

      const currPreset = EXPRESSION_PRESETS[currentIndex];
      const bs = blankBlendshapes();

      if (reElapsed < TRANSITION_DURATION) {
        // ── 过渡阶段：从 stableBs lerp 到 currPreset ──
        const t = Math.min(reElapsed / TRANSITION_DURATION, 1);
        // easeOutCubic：开头快、结尾缓
        const easedT = 1 - Math.pow(1 - t, 3);

        for (const key of Object.keys(bs)) {
          const from = stableBs[key] || 0;
          const to = currPreset.bs[key] || 0;
          bs[key] = from + (to - from) * easedT;
        }

        if (debugOnce) {
          console.log(
            `[LocalArkitsSimulator] 🔄 过渡: ${EXPRESSION_PRESETS[(currentIndex - 1 + EXPRESSION_PRESETS.length) % EXPRESSION_PRESETS.length].name} → ${currPreset.name}`
          );
          debugOnce = false;
        }
      } else {
        // ── 稳定阶段：直接输出目标值 ──
        for (const [key, value] of Object.entries(currPreset.bs)) {
          bs[key] = value;
        }
      }

      // 更新上一帧缓存
      for (const key of Object.keys(prevBs)) {
        prevBs[key] = bs[key];
      }

      frameRef.current = {
        timestamp: Date.now(),
        head: { rotation: { x: 0, y: 0, z: 0 }, position: [0, 0, 0] },
        blendshapes: bs,
      };

      frameIdRef.current = requestAnimationFrame(loop);
    };

    frameIdRef.current = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(frameIdRef.current);
      frameRef.current = null;
      console.log('[LocalArkitsSimulator] ⏹ 已停止');
    };
  }, [enabled, frameRef]);

  return (
    <div
      style={{
        position: 'absolute',
        top: 10,
        left: 10,
        background: '#000000cc',
        color: '#4caf50',
        padding: '8px 14px',
        borderRadius: 8,
        fontSize: '0.82em',
        fontFamily: '"SF Mono", Consolas, monospace',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        lineHeight: 1.4,
      }}
    >
      <span
        style={{
          display: 'inline-block',
          width: 10,
          height: 10,
          borderRadius: '50%',
          background: '#4caf50',
          animation: 'pulse 1.5s infinite',
          flexShrink: 0,
        }}
      />
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <span style={{ fontWeight: 600 }}>🧪 本地模拟模式</span>
        <span style={{ color: '#aaa', fontSize: '0.9em' }}>
          循环: 默认→眨眼→张嘴→大笑→生气→伤心→惊讶→噘嘴→疲惫→嫌弃→鼓脸
        </span>
      </div>
    </div>
  );
};

export default LocalArkitsSimulator;
