import React from 'react';
import { NavLink } from 'react-router-dom';
import { BarChart2, Database, Layers, Settings, Activity, Zap } from 'lucide-react';

const Sidebar: React.FC = () => {
  const navItems = [
    { name: 'Dashboard', path: '/', icon: <BarChart2 className="w-5 h-5" /> },
    { name: 'Datasets', path: '/datasets', icon: <Database className="w-5 h-5" /> },
    { name: 'Fine-Tune', path: '/finetune', icon: <Zap className="w-5 h-5" /> },
    { name: 'Models', path: '/models', icon: <Layers className="w-5 h-5" /> },
    { name: 'Jobs', path: '/jobs', icon: <Activity className="w-5 h-5" /> },
    { name: 'Settings', path: '/settings', icon: <Settings className="w-5 h-5" /> },
  ];

  return (
    <aside className="fixed inset-y-0 left-0 w-64 bg-white border-r border-gray-200 z-10 hidden md:block">
      <div className="h-full flex flex-col">
        <div className="flex-1 overflow-y-auto py-4">
          <nav className="mt-5 px-2 space-y-1">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/'}
                className={({ isActive }) =>
                  `flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-primary-50 text-primary-700'
                      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                  }`
                }
              >
                <span className="mr-3 text-gray-500">{item.icon}</span>
                <span>{item.name}</span>
              </NavLink>
            ))}
          </nav>
        </div>
        <div className="flex-shrink-0 border-t border-gray-200 p-4">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-primary-100">
                <span className="text-sm font-medium leading-none text-primary-700">MA</span>
              </span>
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-700">ModelOps Admin</p>
              <p className="text-xs font-medium text-gray-500">admin@example.com</p>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
