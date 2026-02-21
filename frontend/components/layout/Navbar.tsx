"use client";

import type { User } from "@/types";
import type { View } from "@/app/page";

interface NavbarProps {
    user: User;
    currentView: View;
    onLogout: () => void;
    navigateTo: (view: View) => void;
}

export default function Navbar({ user, currentView, onLogout, navigateTo }: NavbarProps) {
    const initials = user.full_name
        .split(" ")
        .slice(0, 2)
        .map((n) => n[0])
        .join("")
        .toUpperCase();

    const isPhw = user.role === "phw";

    return (
        <nav className="navbar">
            {/* Logo */}
            <button
                className="nav-logo"
                onClick={() => navigateTo("dashboard")}
                style={{ background: "none", border: "none", cursor: "pointer", padding: 0 }}
            >
                <div className="nav-logo-icon">⚕️</div>
                <div>
                    <div style={{ fontSize: "0.95rem", fontWeight: 800, color: "var(--text-primary)" }}>
                        CDSS
                    </div>
                    <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", fontWeight: 500 }}>
                        Clinical Decision Support
                    </div>
                </div>
            </button>

            {/* Nav links */}
            <div className="nav-links">
                <button
                    className={`nav-link ${currentView === "dashboard" ? "active" : ""}`}
                    onClick={() => navigateTo("dashboard")}
                >
                    📊 Dashboard
                </button>
                {isPhw && (
                    <button
                        className={`nav-link ${currentView === "intake" ? "active" : ""}`}
                        onClick={() => navigateTo("intake")}
                    >
                        ➕ New Patient
                    </button>
                )}
                <button className="nav-link" style={{ pointerEvents: "none", opacity: 0.5 }}>
                    📋 Cases
                </button>
            </div>

            {/* User info */}
            <div className="nav-user">
                {/* Live indicator */}
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div
                        style={{
                            width: 8,
                            height: 8,
                            borderRadius: "50%",
                            background: "var(--low)",
                            boxShadow: "0 0 8px var(--low)",
                            animation: "pulse 2s ease-in-out infinite",
                        }}
                    />
                    <span className="text-xs text-muted">Live</span>
                </div>

                <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-primary)" }}>
                        {user.full_name}
                    </div>
                    <div style={{ fontSize: "0.7rem", color: "var(--accent-blue)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                        {user.role === "phw" ? "PHW" : "Specialist"}
                    </div>
                </div>

                <div className="avatar">{initials}</div>

                <button className="btn btn-ghost btn-sm" onClick={onLogout}>
                    Sign out
                </button>
            </div>
        </nav>
    );
}
