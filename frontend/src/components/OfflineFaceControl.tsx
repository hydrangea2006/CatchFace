// src/components/OfflineFaceControl.tsx
// 表情控制 UI 面板（Canvas 外部组件）
// 通过 stateRef 与 Canvas 内部的 VrmModel 通信
import React, { useState, useCallback } from 'react';

export interface FaceControlState {
  weights: Record<string, number>;
}

interface Props {
  /** 模型支持的表情名称列表（来自 VRMExpressionManager） */
  expressionNames: string[];
  /** 通过 ref 将表情参数暴露给 Canvas 内部的 VrmModel */
  stateRef: React.RefObject<FaceControlState>;
}

/** VRM 预设表情的中文映射 */
const PRESET_LABELS: Record<string, string> = {
  neutral: '中立',
  happy: '开心',
  angry: '生气',
  sad: '悲伤',
  relaxed: '放松',
  surprised: '惊讶',
  blink: '眨眼',
  blinkLeft: '左眨眼',
  blinkRight: '右眨眼',
  lookUp: '看上',
  lookDown: '看下',
  lookLeft: '看左',
  lookRight: '看右',
  aa: '嘴型 aa',
  ee: '嘴型 ee',
  ih: '嘴型 ih',
  oh: '嘴型 oh',
  ou: '嘴型 ou',
};

const OfflineFaceControl: React.FC<Props> = ({ expressionNames, stateRef }) => {
  const [weights, setWeights] = useState<Record<string, number>>({});

  // 同步最新 state 到外部 ref（每次渲染更新）
  stateRef.current = { weights };

  const resetAll = useCallback(() => {
    setWeights({});
  }, []);

  if (expressionNames.length === 0) {
    return (
      <div
        style={{
          position: 'absolute',
          top: 10,
          left: 10,
          background: '#ffffffdd',
          padding: 10,
          borderRadius: 8,
          fontSize: '0.85em',
          color: '#888',
        }}
      >
        等待模型加载...
      </div>
    );
  }

  // 将表情名称分组（预设 vs 自定义）
  const presetNames = expressionNames.filter((n) => n in PRESET_LABELS);
  const customNames = expressionNames.filter((n) => !(n in PRESET_LABELS));

  return (
    <div
      style={{
        position: 'absolute',
        top: 10,
        left: 10,
        background: '#ffffffdd',
        padding: '8px 10px',
        borderRadius: 8,
        maxHeight: '80vh',
        overflowY: 'auto',
        width: 220,
        fontSize: '1.2em',
      }}
    >
      <h4 style={{ margin: '0 0 6px' }}>表情控制</h4>
      <small style={{ fontSize: '0.7em', color: '#666' }}>
        VRM 表情 ({expressionNames.length} 个)
      </small>

      {/* 重置按钮 */}
      <div style={{ marginBottom: 8, marginTop: 4 }}>
        <button
          onClick={resetAll}
          style={{
            padding: '4px 8px',
            fontSize: '0.8em',
            cursor: 'pointer',
            background: '#ffe6e6',
            border: '1px solid #cc0000',
            borderRadius: 4,
          }}
        >
          重置全部
        </button>
      </div>

      {/* 预设表情滑块 */}
      {presetNames.length > 0 && (
        <>
          <small style={{ fontSize: '0.7em', color: '#666' }}>预设表情：</small>
          {presetNames.map((name) => (
            <div key={name} style={{ marginBottom: 4 }}>
              <label style={{ fontSize: '0.8em' }}>
                {PRESET_LABELS[name] || name}
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={weights[name] || 0}
                  onChange={(e) =>
                    setWeights((prev) => ({
                      ...prev,
                      [name]: parseFloat(e.target.value),
                    }))
                  }
                  style={{ flex: 1 }}
                />
                <span style={{ fontSize: '0.75em', minWidth: 35 }}>
                  {(weights[name] || 0).toFixed(2)}
                </span>
              </div>
            </div>
          ))}
        </>
      )}

      {/* 自定义表情滑块 */}
      {customNames.length > 0 && (
        <>
          <small style={{ fontSize: '0.7em', color: '#666', marginTop: 8, display: 'block' }}>
            自定义表情：
          </small>
          {customNames.map((name) => (
            <div key={name} style={{ marginBottom: 4 }}>
              <label style={{ fontSize: '0.8em' }}>{name}</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={weights[name] || 0}
                  onChange={(e) =>
                    setWeights((prev) => ({
                      ...prev,
                      [name]: parseFloat(e.target.value),
                    }))
                  }
                  style={{ flex: 1 }}
                />
                <span style={{ fontSize: '0.75em', minWidth: 35 }}>
                  {(weights[name] || 0).toFixed(2)}
                </span>
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  );
};

export default OfflineFaceControl;
