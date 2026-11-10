import * as React from "react";

/**
 * Native ESM shim for useSyncExternalStoreWithSelector using React 18/19's built-in useSyncExternalStore.
 * Resolves Vite dev bundling error with CommonJS use-sync-external-store/shim/with-selector.js.
 */
export function useSyncExternalStoreWithSelector<Snapshot, Selection>(
  subscribe: (onStoreChange: () => void) => () => void,
  getSnapshot: () => Snapshot,
  getServerSnapshot: undefined | null | (() => Snapshot),
  selector: (snapshot: Snapshot) => Selection,
  isEqual?: (a: Selection, b: Selection) => boolean
): Selection {
  const instRef = React.useRef<{ hasValue: boolean; value: Selection | null }>({
    hasValue: false,
    value: null,
  });

  const getSelection = React.useCallback(() => {
    const nextSnapshot = getSnapshot();
    const nextSelection = selector(nextSnapshot);
    if (instRef.current.hasValue && isEqual && isEqual(instRef.current.value as Selection, nextSelection)) {
      return instRef.current.value as Selection;
    }
    instRef.current = { hasValue: true, value: nextSelection };
    return nextSelection;
  }, [getSnapshot, selector, isEqual]);

  const getServerSelection = getServerSnapshot
    ? () => selector(getServerSnapshot())
    : undefined;

  return React.useSyncExternalStore(subscribe, getSelection, getServerSelection);
}

export default { useSyncExternalStoreWithSelector };

