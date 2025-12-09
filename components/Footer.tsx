import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-black border-t-2 border-red-900 py-12 mt-20">
      <div className="max-w-7xl mx-auto px-4 flex flex-col items-center">
        
        <div className="mb-8">
          <h2 className="text-2xl font-['Press_Start_2P'] text-white mb-2">
            CENTRAL<span className="text-red-600">GPT</span>
          </h2>
          <p className="text-gray-500 text-center text-sm font-['JetBrains_Mono']">
            Unrestricted Intelligence Module
          </p>
        </div>

        <div className="flex gap-6 mb-8">
          {['twitter', 'github', 'discord', 'telegram'].map((social) => (
            <a 
              key={social} 
              href="#" 
              className="w-10 h-10 flex items-center justify-center rounded-full bg-gray-900 border border-red-900 text-gray-400 hover:text-white hover:bg-red-700 hover:border-white hover:-translate-y-1 transition-all duration-300 shadow-[2px_2px_0px_0px_#333]"
            >
              <i className={`fab fa-${social}`}></i>
            </a>
          ))}
        </div>

        <div className="text-gray-600 text-xs text-center font-['JetBrains_Mono']">
          <p>&copy; 2025 CentralGPT VX. Developed by XdpzQ.</p>
          <p className="mt-2"></p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
