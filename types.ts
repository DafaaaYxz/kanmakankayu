export interface Message {
  role: 'user' | 'model';
  text: string;
  isError?: boolean;
}

export interface ChatLog {
  id: string;
  userId: string;
  role: 'user' | 'model';
  content: string;
  createdAt: string;
}

export interface StatItem {
  label: string;
  value: string;
  icon: string;
}

export interface NavLink {
  label: string;
  href: string;
  isActive?: boolean;
}

// Admin & Config Types
export interface AppConfig {
  aiName: string;
  aiPersona: string;
  devName: string;
  apiKeys: string[];
  avatarUrl?: string; // Base64 string for the avatar
}

export interface UserProfile {
  fullName: string;
  address: string;
  street: string;
  zipCode: string;
  country: string;
  phoneNumber?: string;
}

export interface User {
  id: string;
  username: string;
  accessKey: string; // Acts as password
  role: 'admin' | 'user';
  createdAt: string;
  profile?: UserProfile; // Optional for admin, required for registered users
  config?: AppConfig; // Specific config for this user/template
}

export interface DatabaseSchema {
  users: User[];
  globalConfig: AppConfig;
}
