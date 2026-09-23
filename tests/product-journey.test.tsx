import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import RelayApp from "../src/components/RelayApp";

describe("product journey", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("turns the required sentence into an editable confirmed session", async () => {
    render(<RelayApp />);
    fireEvent.click(
      screen.getByRole("button", { name: "Create my session" }),
    );

    expect(
      await screen.findByRole("heading", {
        name: "Check what Relay understood.",
      }),
    ).toBeInTheDocument();
    expect(screen.getByDisplayValue("3")).toBeInTheDocument();
    expect(screen.getByDisplayValue("10")).toBeInTheDocument();
    expect(screen.getByDisplayValue("90")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Confirm & set up camera" }),
    );
    expect(
      await screen.findByRole("heading", {
        name: "Let’s find a view Relay can trust.",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "MediaPipe Pose Full + Relay Temporal · video is not saved",
      ),
    ).toBeInTheDocument();
  });
});
