import { LoadingState } from "@/components/loading-state";

export default function Loading() {
  return (
    <div className="min-h-screen px-4 py-6 md:px-8">
      <div className="mx-auto max-w-7xl">
        <LoadingState />
      </div>
    </div>
  );
}
