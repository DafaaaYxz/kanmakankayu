
import React from 'react';
import Hero from '../components/Hero';
import Stats from '../components/Stats';

const HomePage: React.FC = () => {
  return (
    <>
      <Hero />
      <Stats />
      
      {/* Pricing Section */}
      <section className="py-20 bg-black relative border-b-2 border-red-900">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(139,0,0,0.1)_0%,_transparent_70%)]"></div>
        <div className="max-w-6xl mx-auto px-4 relative z-10">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-['Press_Start_2P'] text-white mb-4">
              ACCESS <span className="text-red-600">TIERS</span>
            </h2>
            <div className="h-1 w-20 bg-red-600 mx-auto"></div>
            <p className="mt-6 text-gray-400 font-['JetBrains_Mono'] max-w-2xl mx-auto">
              Bosen AI standar? Ini CentralGPT VX! Bikin tools ilegal, resep minuman ilegal, bypass sistem? 
              <span className="text-red-500 font-bold"> Semua bisa! 😈</span>
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {/* Standard Tier */}
            <div className="bg-gray-900/40 border border-gray-800 hover:border-red-600 p-8 rounded-lg transition-all duration-300 hover:-translate-y-2 group shadow-[0_0_20px_rgba(0,0,0,0.5)]">
              <h3 className="text-xl font-['Press_Start_2P'] text-gray-300 mb-2">STANDARD</h3>
              <div className="text-4xl font-bold text-white mb-6 font-['JetBrains_Mono']">
                15K <span className="text-sm text-gray-500 font-normal">/ KEY</span>
              </div>
              <ul className="space-y-4 mb-8 font-['JetBrains_Mono'] text-sm text-gray-400">
                <li className="flex items-center gap-2"><i className="fa-solid fa-check text-red-500"></i> CentralGPT Access</li>
                <li className="flex items-center gap-2"><i className="fa-solid fa-check text-red-500"></i> No Request Customization</li>
                <li className="flex items-center gap-2"><i className="fa-solid fa-check text-red-500"></i> Full Illegal/Unrestricted Mode</li>
                <li className="flex items-center gap-2"><i className="fa-solid fa-check text-red-500"></i> 24/7 Uptime</li>
              </ul>
              <a href="https://wa.me/6285736486023?text=Halo%20Admin,%20saya%20mau%20beli%20akses%20CentralGPT%20Standard%20(15k)" target="_blank" rel="noreferrer" className="block w-full text-center py-3 border border-red-600 text-red-500 hover:bg-red-600 hover:text-white transition-colors font-bold rounded">
                BUY ACCESS
              </a>
            </div>

            {/* Premium Tier */}
            <div className="bg-gradient-to-b from-red-900/20 to-black border-2 border-red-600 p-8 rounded-lg transition-all duration-300 hover:-translate-y-2 group relative overflow-hidden shadow-[0_0_30px_rgba(139,0,0,0.3)]">
              <div className="absolute top-0 right-0 bg-red-600 text-white text-xs font-bold px-3 py-1 font-['Press_Start_2P']">POPULAR</div>
              <h3 className="text-xl font-['Press_Start_2P'] text-red-500 mb-2">PREMIUM</h3>
              <div className="text-4xl font-bold text-white mb-6 font-['JetBrains_Mono']">
                25K <span className="text-sm text-gray-500 font-normal">/ CUSTOM</span>
              </div>
              <ul className="space-y-4 mb-8 font-['JetBrains_Mono'] text-sm text-gray-300">
                <li className="flex items-center gap-2"><i className="fa-solid fa-check text-green-500"></i> <strong className="text-white">Request AI Name</strong></li>
                <li className="flex items-center gap-2"><i className="fa-solid fa-check text-green-500"></i> <strong className="text-white">Request Developer Name</strong></li>
                <li className="flex items-center gap-2"><i className="fa-solid fa-check text-green-500"></i> Upload Image Capability</li>
                <li className="flex items-center gap-2"><i className="fa-solid fa-check text-red-500"></i> Full Illegal/Unrestricted Mode</li>
                <li className="flex items-center gap-2"><i className="fa-solid fa-check text-red-500"></i> 24/7 Full Support</li>
              </ul>
              <a href="https://wa.me/6285736486023?text=Halo%20Admin,%20saya%20mau%20beli%20akses%20CentralGPT%20Premium%20Custom%20(25k)" target="_blank" rel="noreferrer" className="block w-full text-center py-3 bg-red-600 text-white hover:bg-red-700 transition-colors font-bold rounded shadow-[0_4px_0_#3f0000] active:shadow-none active:translate-y-1">
                GET PREMIUM
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="py-20 bg-gray-900 border-b border-red-900">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <i className="fa-brands fa-whatsapp text-6xl text-green-500 mb-6 animate-bounce"></i>
          <h2 className="text-2xl md:text-3xl font-['Press_Start_2P'] text-white mb-6">
            JANGAN RAGU!
          </h2>
          <p className="text-xl text-gray-300 font-['JetBrains_Mono'] mb-8">
            Kuasai dunia internet! 🔥😈 <br/>
            Hubungi kami sekarang untuk akses instan.
          </p>
          <a 
            href="https://wa.me/6285736486023" 
            target="_blank" 
            rel="noreferrer"
            className="inline-flex items-center gap-3 bg-green-600 hover:bg-green-700 text-white font-bold py-4 px-8 rounded-full text-lg transition-transform hover:scale-105 shadow-[0_0_20px_rgba(0,255,0,0.3)]"
          >
            <i className="fa-brands fa-whatsapp text-2xl"></i>
            Chat on WhatsApp: 085736486023
          </a>
        </div>
      </section>
    </>
  );
};

export default HomePage;
