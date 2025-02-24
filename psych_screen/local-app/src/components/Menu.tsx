import React, { useState } from "react";
import DropDown from "./DropDown";

const Menu: React.FC<{ sampleSelection: (sample: string) => void }> = ({sampleSelection}): JSX.Element => {
  const [showDropDown, setShowDropDown] = useState<boolean>(false);
  const [selectedSample, setSelectedSample] = useState<string>("DLPFC_Br8667_mid_manual_alignment_all");
  const samples = () => {
    return [
        "DLPFC_Br8667_mid_manual_alignment_all",
        "DLPFC_Br2720_ant_2",
        "DLPFC_Br2743_post_manual_alignment",
        "DLPFC_Br6471_ant_manual_alignment_all"
    ];
  };

  /**
   * Toggle the drop down menu
   */
  const toggleDropDown = () => {
    setShowDropDown(!showDropDown);
  };

  /**
   * Hide the drop down menu if click occurs
   * outside of the drop-down element.
   *
   * @param event  The mouse event
   */
  const dismissHandler = (event: React.FocusEvent<HTMLButtonElement>): void => {
    if (event.currentTarget === event.target) {
      setShowDropDown(false);
    }
  };

  /**
   * Callback function to consume the
   * sample name from the child component
   * AND send back to parent component
   *
   * @param sample  The selected sample
   */
  const handleSampleSelection = (sample: string): void => {
    setSelectedSample(sample);
    sampleSelection(sample)
  };

  return (
    <>
      <div className="announcement">
        <div>
          {selectedSample
            ? `You selected ${selectedSample} to visualize`
            : "Select your visualization"}
        </div>
      </div>
      <button
        className={showDropDown ? "active" : undefined}
        onClick={(): void => toggleDropDown()}
        onBlur={(e: React.FocusEvent<HTMLButtonElement>): void =>
          dismissHandler(e)
        }
      >
        <div>{selectedSample ? "Select: " + selectedSample : "Select ..."} </div>
        {showDropDown && (
          <DropDown
            samples={samples()}
            showDropDown={false}
            toggleDropDown={(): void => toggleDropDown()}
            sampleSelection={handleSampleSelection}
          />
        )}
      </button>
    </>
  );
};

export default Menu;
