
import React, { useEffect, useState } from 'react';

const LoadingScreen: React.FC<{ onComplete: () => void }> = ({ onComplete }) => {
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  
  const systemChecks = [
    "INITIALIZING KERNEL...",
    "LOADING NEURAL MODULES...",
    "BYPASSING SECURITY PROTOCOLS...",
    "CONNECTING TO CENTRAL NETWORK...",
    "DECRYPTING USER DATA...",
    "ACCESS GRANTED."
  ];

  useEffect(() => {
    // Progress Bar Animation
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(onComplete, 800); // Small delay before unmount
          return 100;
        }
        // Random increment for realistic effect
        return prev + Math.floor(Math.random() * 10) + 1; 
      });
    }, 200);

    return () => clearInterval(interval);
  }, [onComplete]);

  // Log scrolling animation
  useEffect(() => {
    let logIndex = 0;
    const logInterval = setInterval(() => {
      if (logIndex < systemChecks.length) {
        setLogs(prev => [...prev, systemChecks[logIndex]]);
        logIndex++;
      } else {
        clearInterval(logInterval);
      }
    }, 400);

    return () => clearInterval(logInterval);
  }, []);

  return (
    <div className="fixed inset-0 bg-black z-[100] flex flex-col items-center justify-center text-red-600 font-['JetBrains_Mono'] overflow-hidden">
      {/* Background Matrix Effect (Simplified) */}
      <div className="absolute inset-0 opacity-10 pointer-events-none bg-[url('https://media.giphy.com/media/U3qYN8S0j3bpK/giphy.gif')] bg-cover"></div>
      
      <div className="relative z-10 w-full max-w-md p-8">
        <h1 className="text-3xl md:text-5xl font-['Press_Start_2P'] text-center mb-8 animate-pulse text-white">
          CENTRAL<span className="text-red-600">GPT</span>
        </h1>

        {/* Terminal Output */}
        <div className="h-32 bg-gray-900/50 border border-red-900/50 p-4 mb-8 rounded font-mono text-xs text-red-400 overflow-hidden flex flex-col justify-end shadow-[0_0_20px_rgba(139,0,0,0.2)]">
          {logs.map((log, idx) => (
            <div key={idx} className="mb-1">
              <span className="text-gray-500 mr-2">[{new Date().toLocaleTimeString()}]</span>
              <span className={idx === logs.length - 1 ? "text-white animate-pulse" : ""}>{log}</span>
            </div>
          ))}
        </div>

        {/* Progress Bar */}
        <div className="relative h-2 bg-gray-900 rounded overflow-hidden border border-red-900">
          <div 
            className="absolute top-0 left-0 h-full bg-red-600 transition-all duration-200 ease-out shadow-[0_0_10px_red]"
            style={{ width: `${Math.min(progress, 100)}%` }}
          ></div>
        </div>
        <div className="flex justify-between mt-2 text-xs font-bold font-['Press_Start_2P']">
          <span>SYSTEM_BOOT</span>
          <span>{Math.min(progress, 100)}%</span>
        </div>
      </div>
    </div>
  );
};

export default LoadingScreen;
