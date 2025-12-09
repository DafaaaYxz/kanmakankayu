import React from 'react';
import { StatItem } from '../types';

const Stats: React.FC = () => {
  const stats: StatItem[] = [
    { label: "Uptime", value: "99.9%", icon: "fa-clock" },
    { label: "Users", value: "8.2K", icon: "fa-users" },
    { label: "Latency", value: "12ms", icon: "fa-bolt" },
    { label: "Threats", value: "0", icon: "fa-shield-halved" },
  ];

  return (
    <div className="py-12 bg-black relative">
      <div className="max-w-7xl mx-auto px-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <div 
              key={index}
              className="bg-gray-900/50 border-2 border-red-900 p-6 text-center relative group hover:bg-red-900/10 transition-colors duration-300"
              style={{
                boxShadow: '4px 4px 0px 0px rgba(139, 0, 0, 0.4)'
              }}
            >
              {/* Corner Accents */}
              <div className="absolute top-0 left-0 w-2 h-2 bg-red-600"></div>
              <div className="absolute top-0 right-0 w-2 h-2 bg-red-600"></div>
              <div className="absolute bottom-0 left-0 w-2 h-2 bg-red-600"></div>
              <div className="absolute bottom-0 right-0 w-2 h-2 bg-red-600"></div>

              <i className={`fa-solid ${stat.icon} text-3xl text-red-500 mb-4 group-hover:scale-110 transition-transform duration-300`}></i>
              <div className="font-['Press_Start_2P'] text-xl text-white mb-2 shadow-red-500 drop-shadow-md">
                {stat.value}
              </div>
              <div className="text-gray-500 text-xs uppercase tracking-widest font-bold">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Stats;