import React from 'react';
import { APP_NAME, APP_VERSION, DEV_INFO } from '../constants';

const AboutPage: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto px-4 py-12 min-h-[80vh] flex items-center">
      <div className="w-full bg-black/80 border-2 border-red-900 p-8 md:p-12 relative overflow-hidden shadow-[0_0_50px_rgba(139,0,0,0.2)]">
         
         {/* Decorative Elements */}
         <div className="absolute top-0 right-0 bg-red-600 text-white text-xs px-3 py-1 font-bold font-['JetBrains_Mono']">CONFIDENTIAL</div>
         <div className="absolute -left-10 top-1/2 w-20 h-1 bg-red-900 transform -rotate-45"></div>
         <div className="absolute -right-10 bottom-1/2 w-20 h-1 bg-red-900 transform -rotate-45"></div>

         <h1 className="text-3xl md:text-5xl font-['Press_Start_2P'] text-white mb-8 relative z-10">
           ABOUT <span className="text-red-600">{APP_NAME}</span>
         </h1>
         
         <div className="space-y-8 text-gray-300 font-['JetBrains_Mono'] leading-relaxed relative z-10">
           <div className="flex flex-col md:flex-row gap-8 border-b border-gray-800 pb-8">
             <div className="flex-1">
               <p className="mb-2"><strong className="text-red-500">SYSTEM VERSION:</strong> {APP_VERSION}</p>
               <p className="mb-2"><strong className="text-red-500">BUILD IDENTIFIER:</strong> VX-2025-ALPHA</p>
               <p className="mb-2"><strong className="text-red-500">SECURITY CLEARANCE:</strong> LEVEL 5</p>
             </div>
             <div className="flex-1 bg-red-900/10 p-4 border-l-4 border-red-600 italic">
               "{DEV_INFO}"
             </div>
           </div>

           <div>
             <h3 className="text-xl text-white mb-6 font-bold flex items-center gap-2">
               <i className="fas fa-layer-group text-red-500"></i> TECHNOLOGY STACK
             </h3>
             <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
               {[
                 { icon: 'fa-react', name: 'React 18', color: 'text-blue-400' },
                 { icon: 'fa-google', name: 'Gemini', color: 'text-yellow-400' },
                 { icon: 'fa-wind', name: 'Tailwind', color: 'text-cyan-400' },
                 { icon: 'fa-code', name: 'TypeScript', color: 'text-blue-600' },
                 { icon: 'fa-route', name: 'Router', color: 'text-red-400' },
                 { icon: 'fa-wand-magic-sparkles', name: 'Pixel Art', color: 'text-purple-400' }
               ].map((tech, idx) => (
                 <div key={idx} className="flex flex-col items-center justify-center p-4 bg-gray-900/50 border border-gray-800 hover:border-red-600 transition-all hover:-translate-y-1 rounded">
                   <i className={`fab ${tech.icon} ${tech.name === 'Tailwind' || tech.name === 'Router' || tech.name === 'Pixel Art' ? 'fas' : ''} ${tech.color} text-2xl mb-2`}></i>
                   <span className="text-xs font-bold">{tech.name}</span>
                 </div>
               ))}
             </div>
           </div>
         </div>

         <div className="mt-8 text-center text-xs text-gray-600 pt-8 border-t border-gray-800">
            SYSTEM GENERATED MESSAGE: END OF FILE
         </div>
      </div>
    </div>
  );
};

export default AboutPage;
