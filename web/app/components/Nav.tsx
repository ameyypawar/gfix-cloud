"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav className="nav">
      <div className="nav-inner">
        <span className="nav-brand">gfix-cloud</span>
        <ul className="nav-links">
          <li>
            <Link href="/" className={pathname === "/" ? "active" : ""}>
              Resolve
            </Link>
          </li>
          <li>
            <Link
              href="/eval"
              className={pathname?.startsWith("/eval") ? "active" : ""}
            >
              Eval
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}
