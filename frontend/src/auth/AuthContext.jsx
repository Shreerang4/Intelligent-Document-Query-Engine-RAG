import { createContext, useContext, useEffect, useMemo, useSyncExternalStore } from 'react';
import { browserAuthSession } from './authSession.js';


const AuthContext = createContext(null);

export function AuthProvider({ children, session = browserAuthSession }) {
  const snapshot = useSyncExternalStore(
    session.subscribe,
    session.getSnapshot,
    session.getSnapshot,
  );

  useEffect(() => {
    void session.bootstrap();
  }, [session]);

  const value = useMemo(() => ({
    ...snapshot,
    register: session.register,
    login: session.login,
    refresh: session.refresh,
    logout: session.logout,
    authenticatedFetch: session.authenticatedFetch,
    getCurrentUser: session.getCurrentUser,
  }), [session, snapshot]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used inside AuthProvider.');
  }
  return context;
}
