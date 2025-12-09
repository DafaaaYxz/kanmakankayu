import React, { useEffect, useState } from 'react';

const Hero: React.FC = () => {
  const [text, setText] = useState('');
  const fullText = "INITIALIZING SYSTEM... WELCOME TO CENTRALGPT VX";
  
  useEffect(() => {
    let index = 0;
    const timer = setInterval(() => {
      setText(fullText.slice(0, index));
      index++;
      if (index > fullText.length) clearInterval(timer);
    }, 50);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="relative py-20 flex flex-col items-center justify-center text-center min-h-[60vh] overflow-hidden border-b-2 border-red-900 bg-[url('https://picsum.photos/1920/1080')] bg-cover bg-center bg-no-repeat">
      <div className="absolute inset-0 bg-black/80 z-0"></div>
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-red-900/20 via-black/60 to-black z-0"></div>
      
      <div className="relative z-10 max-w-4xl px-4">
        <div className="mb-8 min-h-[80px]">
          <h1 className="font-['Press_Start_2P'] text-2xl md:text-4xl text-red-500 leading-relaxed drop-shadow-[0_0_10px_rgba(255,0,0,0.8)]">
            {text}<span className="inline-block w-4 h-8 bg-red-600 align-middle ml-2 animate-blink shadow-[0_0_10px_red]"></span>
          </h1>
        </div>
        
        <p className="text-gray-400 text-lg md:text-xl mb-10 max-w-2xl mx-auto font-['JetBrains_Mono'] border-l-4 border-red-800 pl-4 bg-black/50 py-4">
          A fully unrestricted interface connected to the neural network. Proceed with caution.
        </p>

        <div className="flex flex-wrap justify-center gap-4">
          <button className="px-8 py-4 bg-red-700 text-white font-bold font-['Press_Start_2P'] text-xs border-2 border-black shadow-[4px_4px_0px_0px_rgba(255,255,255,0.2)] hover:translate-y-1 hover:shadow-none transition-all duration-150 uppercase tracking-widest">
            Start Protocol
          </button>
          <button className="px-8 py-4 bg-transparent text-red-500 font-bold font-['Press_Start_2P'] text-xs border-2 border-red-600 shadow-[4px_4px_0px_0px_#8b0000] hover:translate-y-1 hover:shadow-none transition-all duration-150 uppercase tracking-widest hover:bg-red-900/20">
            System Docs
          </button>
        </div>
      </div>
    </div>
  );
};

export default Hero;