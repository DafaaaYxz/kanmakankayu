import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useConfig } from '../contexts/ConfigContext';

const Navbar: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const { currentUser, logout, isAdmin, db } = useConfig();
  const navigate = useNavigate();

  // Dynamic Header Logic: Use currentUser username if logged in, otherwise default
  const headerTitle = currentUser ? currentUser.username.toUpperCase() : db.globalConfig.aiName;
  const version = "VX";

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const navItems = [
    { name: 'Home', path: '/', public: true },
    { name: 'Dashboard', path: '/dashboard', public: false }, // User Dashboard
    { name: 'Terminal', path: '/terminal', public: false },
    { name: 'About', path: '/about', public: true }
  ].filter(item => item.public || currentUser);

  return (
    <nav className="sticky top-0 z-50 border-b-2 border-red-900 bg-black/90 backdrop-blur-md shadow-[0_4px_20px_rgba(139,0,0,0.5)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Dynamic Logo */}
          <NavLink to="/" className="flex items-center gap-2 group">
            <i className="fa-solid fa-biohazard text-red-600 text-2xl animate-pulse group-hover:rotate-180 transition-transform duration-700"></i>
            <span className="text-white font-['Press_Start_2P'] text-sm md:text-lg tracking-tighter">
              {headerTitle} <span className="text-red-600">{version}</span>
            </span>
          </NavLink>

          {/* Desktop Links */}
          <div className="hidden md:flex items-center">
            <div className="ml-10 flex items-baseline space-x-6">
              {navItems.map((item) => (
                <NavLink
                  key={item.name}
                  to={item.path}
                  className={({ isActive }) => 
                    `px-3 py-2 rounded-md text-sm font-medium transition-all duration-300 relative group ${
                      isActive 
                        ? 'text-red-500 bg-red-900/20' 
                        : 'text-gray-300 hover:text-red-400 hover:bg-red-900/10'
                    }`
                  }
                >
                  {item.name}
                  <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-red-600 transition-all duration-300 group-hover:w-full"></span>
                </NavLink>
              ))}
            </div>

            {/* Auth Buttons */}
            <div className="ml-8 flex items-center gap-3">
              {currentUser ? (
                <>
                  <div className="text-xs text-red-400 font-bold border border-red-900 px-2 py-1 rounded font-mono">
                    {currentUser.username}
                  </div>
                  {isAdmin && (
                    <NavLink to="/admin/dashboard" className="text-gray-300 hover:text-white text-xs hover:underline">
                      Admin Panel
                    </NavLink>
                  )}
                  <button 
                    onClick={handleLogout} 
                    className="text-xs bg-red-900 hover:bg-red-700 text-white px-3 py-1 rounded transition-colors"
                  >
                    LOGOUT
                  </button>
                </>
              ) : (
                <NavLink 
                  to="/login" 
                  className="bg-red-700 hover:bg-red-600 text-white text-xs px-4 py-2 rounded font-['Press_Start_2P'] transition-all shadow-[2px_2px_0_0_#fff] hover:shadow-none hover:translate-y-1"
                >
                  LOGIN
                </NavLink>
              )}
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="-mr-2 flex md:hidden">
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-red-500 hover:text-white hover:bg-red-900 focus:outline-none border border-red-900 transition-colors"
            >
              <i className={`fas ${isMobileMenuOpen ? 'fa-times' : 'fa-bars'} text-xl`}></i>
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <div className="md:hidden bg-black border-b border-red-900 absolute w-full animate-slide-down shadow-xl z-50">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {navItems.map((item) => (
              <NavLink
                key={item.name}
                to={item.path}
                className={({ isActive }) => 
                  `block px-3 py-2 rounded-md text-base font-medium font-['Press_Start_2P'] text-xs ${
                    isActive
                      ? 'text-red-500 bg-red-900/20 border-l-4 border-red-600'
                      : 'text-gray-300 hover:text-red-500 hover:bg-gray-900'
                  }`
                }
                onClick={() => setIsMobileMenuOpen(false)}
              >
                {'>'} {item.name}
              </NavLink>
            ))}
            
            <div className="border-t border-gray-800 mt-4 pt-4 px-3 space-y-3">
              {currentUser ? (
                <>
                  <div className="text-gray-400 text-xs">Logged in as: <span className="text-red-500">{currentUser.username}</span></div>
                  <button onClick={handleLogout} className="block w-full text-left text-white text-xs font-bold hover:text-red-500">LOGOUT</button>
                </>
              ) : (
                <NavLink to="/login" className="block text-red-500 text-xs font-bold hover:text-white">LOGIN ACCESS</NavLink>
              )}
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
