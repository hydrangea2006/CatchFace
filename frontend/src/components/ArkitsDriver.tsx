// src/components/ArkitsDriver.tsx
// ARkit 面部数据驱动组件
// - 通过 WebSocket (socket.io) 接收后端推送的 blendshapes + 头部姿态
// - 将数据写入共享 ref，供 VrmModel 每帧消费
// - 显示连接状态
import React, { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import type { ArkitsFrame } from './VrmModel';

interface Props {
  /** 后端 WebSocket 地址，例如 "http://localhost:5000" */
  serverUrl: string;
  /** 共享引用：ArkitsDriver 写入 → VrmModel useFrame 读取 */
  frameRef: React.RefObject<ArkitsFrame | null>;
}

const ArkitsDriver: React.FC<Props> = ({ serverUrl, frameRef }) => {
  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    const socket: Socket = io(serverUrl, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('[ArkitsDriver] 已连接:', serverUrl);
      setStatus('connected');
    });

    // 接收后端推送的面部数据帧
    socket.on('face_data', (data: ArkitsFrame) => {
      // 直接写入 ref，不触发 React 重渲染（高频数据 30~60fps）
      frameRef.current = data;
    });

    socket.on('disconnect', (reason) => {
      console.warn('[ArkitsDriver] 断开:', reason);
      setStatus('disconnected');
    });

    socket.on('connect_error', (err) => {
      console.error('[ArkitsDriver] 连接错误:', err.message);
      setStatus('disconnected');
    });

    return () => {
      socket.removeAllListeners();
      socket.disconnect();
    };
  }, [serverUrl, frameRef]);

  return (
    <div
      style={{
        position: 'absolute',
        top: 10,
        right: 10,
        background: '#000000aa',
        color: '#fff',
        padding: '6px 12px',
        borderRadius: 6,
        fontSize: '0.8em',
        fontFamily: 'sans-serif',
        display: 'flex',
        alignItems: 'center',
        gap: 6,
      }}
    >
      <span
        style={{
          display: 'inline-block',
          width: 8,
          height: 8,
          borderRadius: '50%',
          background:
            status === 'connected'
              ? '#4caf50'
              : status === 'connecting'
                ? '#ff9800'
                : '#f44336',
        }}
      />
      {status === 'connected'
        ? '面部追踪已连接'
        : status === 'connecting'
          ? '连接中...'
          : '未连接'}
    </div>
  );
};

export default ArkitsDriver;
