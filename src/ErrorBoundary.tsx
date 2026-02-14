import React from 'react';

export default class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: unknown) {
    console.error('Lung GLB viewer error:', error);
  }

  render() {
    if (this.state.hasError) return null;
    return this.props.children;
  }
}
