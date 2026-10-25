import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  /** shown in the console so a stack trace says which area failed */
  area: string;
  /** boundary resets itself when this changes (e.g. switching tabs) */
  resetKey?: string;
}
interface State {
  error: Error | null;
}

/** Keeps one broken screen from blanking the whole page. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[recall] ${this.props.area} crashed:`, error, info.componentStack);
  }

  componentDidUpdate(prev: Props) {
    if (this.state.error && prev.resetKey !== this.props.resetKey) this.setState({ error: null });
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className="mx-auto w-full max-w-[420px] rounded-r20 bg-card p-4 text-center text-label">
        <div className="t-headline">Something went wrong</div>
        <p className="t-subhead mb-4 mt-1 text-label-2">This screen hit a problem. Reloading usually fixes it.</p>
        <button
          onClick={() => window.location.reload()}
          className="t-headline pressable w-full rounded-r14 bg-label py-3 text-on-label"
        >
          Reload
        </button>
      </div>
    );
  }
}
